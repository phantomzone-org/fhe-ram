use core::{
    AutomorphismKey, GLWECiphertext, GLWEOps, GLWEPlaintext, ScratchCore, SecretKey,
    SecretKeyFourier, StreamPacker, TensorKey,
};
use std::collections::HashMap;

use backend::{Encoding, FFT64, Module, Scratch, ScratchOwned};
use itertools::izip;
use sampling::source::{Source, new_seed};

use crate::{
    address::{Address, Coordinate},
    keys::EvaluationKeys,
    parameters::Parameters,
    reverse_bits_msb,
};

/// [Ram] core implementation of the FHE-RAM.
pub struct Ram {
    pub(crate) params: Parameters,
    pub(crate) subrams: Vec<SubRam>,
    pub(crate) scratch: ScratchOwned,
}

impl Ram {
    /// Instantiates a new [Ram].
    pub fn new() -> Self {
        let params: Parameters = Parameters::new();
        let scratch: ScratchOwned = ScratchOwned::new(Self::scratch_bytes(&params));

        let mut subrams: Vec<SubRam> = Vec::new();
        (0..params.word_size()).for_each(|_| {
            subrams.push(SubRam::alloc(&params));
        });

        Self {
            subrams,
            params,
            scratch,
        }
    }

    /// Scratch space size required by the [Ram].
    pub(crate) fn scratch_bytes(params: &Parameters) -> usize {
        let module: &Module<FFT64> = params.module();
        let k_ct: usize = params.k_ct();
        let k_evk: usize = params.k_evk();
        let basek: usize = params.basek();
        let rank: usize = params.rank();

        let enc_sk: usize = GLWECiphertext::encrypt_sk_scratch_space(module, basek, k_ct);
        let coordinate_product: usize = Coordinate::product_scratch_space(params);
        let packing: usize = StreamPacker::scratch_space(module, basek, k_ct, k_evk, rank);
        let trace: usize =
            GLWECiphertext::trace_inplace_scratch_space(module, basek, k_ct, k_evk, rank);
        let ct: usize = GLWECiphertext::bytes_of(module, basek, k_ct, rank);
        let inv_addr: usize = Coordinate::invert_scratch_space(params);

        let read: usize = coordinate_product | trace | packing;
        let write: usize = coordinate_product | ct + trace | inv_addr;

        enc_sk | read | write
    }

    /// Initialize the FHE-[Ram] with provided values (encrypted inder the provided secret).
    pub fn encrypt_sk(&mut self, data: &[u8], sk: &SecretKey<Vec<u8>>) {
        let params: &Parameters = &self.params;
        let max_addr: usize = params.max_addr();
        let ram_chunks: usize = params.word_size();

        assert!(
            data.len() % ram_chunks == 0,
            "invalid data: data.len()%ram_chunks={} != 0",
            data.len() % ram_chunks,
        );

        assert!(
            data.len() / ram_chunks == max_addr,
            "invalid data: data.len()/ram_chunks={} != max_addr={}",
            data.len() / ram_chunks,
            max_addr
        );

        let module: &Module<FFT64> = &params.module();
        let scratch: &mut Scratch = self.scratch.borrow();
        let rank: usize = params.rank();

        let mut sk_dft: SecretKeyFourier<Vec<u8>, FFT64> = SecretKeyFourier::alloc(module, rank);
        sk_dft.dft(module, sk);

        let mut data_split: Vec<u8> = vec![0u8; max_addr];
        (0..ram_chunks).for_each(|i| {
            data_split.iter_mut().enumerate().for_each(|(j, x)| {
                *x = data[j + i];
            });
            self.subrams[i].encrypt_sk(params, &data_split, &sk_dft, scratch);
        });
    }

    /// Simple read from the [Ram] at the provided encrypted address.
    /// Returns a vector of [GLWECiphertext], where each ciphertext stores
    /// Enc(m_i) where is the i-th digit of the word-size such that m = m_0 | m-1 | ...
    pub fn read(
        &mut self,
        address: &Address,
        keys: &EvaluationKeys,
    ) -> Vec<GLWECiphertext<Vec<u8>>> {
        assert!(
            self.subrams.len() != 0,
            "unitialized memory: self.data.len()=0"
        );

        let mut res: Vec<GLWECiphertext<Vec<u8>>> = Vec::new();
        self.subrams.iter_mut().for_each(|subram| {
            res.push(subram.read(
                &self.params,
                address,
                &keys.auto_keys,
                self.scratch.borrow(),
            ))
        });

        res
    }

    /// Read that prepares the [Ram] of a subsequent [Self::write].
    /// Outside of preparing the [Ram] for a write, the behavior and
    /// output format is identical to [Self::read].
    pub fn read_prepare_write(
        &mut self,
        address: &Address,
        keys: &EvaluationKeys,
    ) -> Vec<GLWECiphertext<Vec<u8>>> {
        assert!(
            self.subrams.len() != 0,
            "unitialized memory: self.data.len()=0"
        );

        let mut res: Vec<GLWECiphertext<Vec<u8>>> = Vec::new();
        self.subrams.iter_mut().for_each(|subram| {
            res.push(subram.read_prepare_write(
                &self.params,
                address,
                &keys.auto_keys,
                self.scratch.borrow(),
            ))
        });

        res
    }

    /// Writes w to the [Ram]. Requires that [Self::read_prepare_write] was
    /// called beforehand.
    pub fn write<DataW: AsRef<[u8]>>(
        &mut self,
        w: &Vec<GLWECiphertext<DataW>>, // Must encrypt [w, 0, 0, ..., 0];
        address: &Address,
        keys: &EvaluationKeys,
    ) {
        assert!(w.len() == self.subrams.len());

        let params: &Parameters = &self.params;
        let module: &Module<FFT64> = &params.module();
        let basek: usize = params.basek();
        let rank: usize = params.rank();

        let scratch: &mut Scratch = self.scratch.borrow();
        let auto_keys: &HashMap<i64, AutomorphismKey<Vec<u8>, FFT64>> = &keys.auto_keys;
        let tensor_key: &TensorKey<Vec<u8>, FFT64> = &keys.tensor_key;

        // Overwrites the coefficient that was read: to_write_on = to_write_on - TRACE(to_write_on) + w
        self.subrams.iter_mut().enumerate().for_each(|(i, subram)| {
            subram.write_first_step(params, &w[i], address.n2(), auto_keys, scratch);
        });

        for i in (0..address.n2() - 1).rev() {
            // Index polynomial X^{i}
            let coordinate: &Coordinate<Vec<u8>> = address.at(i + 1);

            let mut inv_coordinate: Coordinate<Vec<u8>> = Coordinate::alloc(
                module,
                basek,
                address.k(),
                address.rows(),
                rank,
                &coordinate.base1d.clone(),
            ); // DODO ALLOC FROM SCRATCH SPACE

            inv_coordinate.base1d = coordinate.base1d.clone();

            // Inverts coordinate: X^{i} -> X^{-i}
            inv_coordinate.invert(
                module,
                coordinate,
                auto_keys.get(&-1).unwrap(),
                tensor_key,
                scratch,
            );

            self.subrams.iter_mut().for_each(|subram| {
                subram.write_mid_step(i, params, &inv_coordinate, auto_keys, scratch);
            });
        }

        let coordinate: &Coordinate<Vec<u8>> = address.at(0);

        let mut inv_coordinate: Coordinate<Vec<u8>> = Coordinate::alloc(
            module,
            basek,
            address.k(),
            address.rows(),
            rank,
            &coordinate.base1d.clone(),
        ); // DODO ALLOC FROM SCRATCH SPACE

        // Inverts coordinate: X^{i} -> X^{-i}
        inv_coordinate.invert(
            module,
            coordinate,
            auto_keys.get(&-1).unwrap(),
            tensor_key,
            scratch,
        );

        self.subrams.iter_mut().for_each(|subram| {
            subram.write_last_step(module, &inv_coordinate, scratch);
        })
    }
}

/// [SubRam] stores a digit of the word.
pub(crate) struct SubRam {
    data: Vec<GLWECiphertext<Vec<u8>>>,
    tree: Vec<Vec<GLWECiphertext<Vec<u8>>>>,
    packer: StreamPacker,
    state: bool,
}

impl SubRam {
    pub fn alloc(params: &Parameters) -> Self {
        let module: &Module<FFT64> = &params.module();
        let basek: usize = params.basek();
        let k_ct: usize = params.k_ct();
        let rank: usize = params.rank();

        let n: usize = module.n();
        let mut tree: Vec<Vec<GLWECiphertext<Vec<u8>>>> = Vec::new();
        let max_addr_split: usize = params.max_addr() >> 2; // u8 -> u32

        if max_addr_split > n {
            let mut size: usize = (max_addr_split + n - 1) / n;
            while size != 1 {
                size = (size + n - 1) / n;
                let mut tmp: Vec<GLWECiphertext<Vec<u8>>> = Vec::new();
                (0..size).for_each(|_| tmp.push(GLWECiphertext::alloc(module, basek, k_ct, rank)));
                tree.push(tmp);
            }
        }

        Self {
            data: Vec::new(),
            tree: tree,
            packer: StreamPacker::new(module, 0, basek, k_ct, rank),
            state: false,
        }
    }

    pub fn encrypt_sk(
        &mut self,
        params: &Parameters,
        data: &[u8],
        sk_dft: &SecretKeyFourier<Vec<u8>, FFT64>,
        scratch: &mut Scratch,
    ) {
        let module: &Module<FFT64> = &params.module();
        let k_pt: usize = params.k_pt();
        let sigma: f64 = params.xe();
        let rank: usize = params.rank();
        let basek: usize = params.basek();
        let k_ct: usize = params.k_ct();

        let mut source_xa: Source = Source::new(new_seed());
        let mut source_xe: Source = Source::new(new_seed());

        let cts: &mut Vec<GLWECiphertext<Vec<u8>>> = &mut self.data;
        let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(module, basek, k_pt);
        let mut data_i64: Vec<i64> = vec![0i64; module.n()];

        for chunk in data.chunks(module.n()) {
            let mut ct: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(module, basek, k_ct, rank);
            izip!(data_i64.iter_mut(), chunk.iter()).for_each(|(xi64, xu8)| *xi64 = *xu8 as i64);
            data_i64[chunk.len()..].iter_mut().for_each(|x| *x = 0);
            pt.data.encode_vec_i64(0, basek, k_pt, &data_i64, 8);
            ct.encrypt_sk::<_, Vec<u8>>(
                module,
                &pt,
                &sk_dft,
                &mut source_xa,
                &mut source_xe,
                sigma,
                scratch,
            );
            cts.push(ct);
        }
    }

    fn read(
        &mut self,
        params: &Parameters,
        address: &Address,
        auto_keys: &HashMap<i64, AutomorphismKey<Vec<u8>, FFT64>>,
        scratch: &mut Scratch,
    ) -> GLWECiphertext<Vec<u8>> {
        assert_eq!(
            self.state, false,
            "invalid call to Memory.read: internal state is true -> requires calling Memory.write"
        );

        let module: &Module<FFT64> = &params.module();
        let log_n: usize = module.log_n();
        let basek: usize = params.basek();
        let k_ct: usize = params.k_ct();
        let rank: usize = params.rank();
        let packer: &mut StreamPacker = &mut self.packer;

        let mut results: Vec<GLWECiphertext<Vec<u8>>> = Vec::new();
        let mut tmp_ct: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(module, basek, k_ct, rank);

        for i in 0..address.n2() {
            let coordinate: &Coordinate<Vec<u8>> = address.at(i);

            let res_prev: &Vec<GLWECiphertext<Vec<u8>>>;

            if i == 0 {
                res_prev = &self.data
            } else {
                res_prev = &results
            }

            if i < address.n2() - 1 {
                let mut result_next: Vec<GLWECiphertext<Vec<u8>>> = Vec::new();

                for chunk in res_prev.chunks(module.n()) {
                    (0..module.n()).for_each(|j| {
                        let j_rev = reverse_bits_msb(j, log_n as u32);

                        if j_rev < chunk.len() {
                            coordinate.product(module, &mut tmp_ct, &chunk[j_rev], scratch);
                            packer.add(module, &mut result_next, Some(&tmp_ct), auto_keys, scratch);
                        } else {
                            packer.add(
                                module,
                                &mut result_next,
                                None::<&GLWECiphertext<Vec<u8>>>,
                                auto_keys,
                                scratch,
                            );
                        }
                    });
                }

                packer.flush(module, &mut result_next, auto_keys, scratch);
                packer.reset();
                results = result_next;
            } else {
                if i == 0 {
                    coordinate.product(module, &mut tmp_ct, &self.data[0], scratch);
                } else {
                    coordinate.product(module, &mut tmp_ct, &results[0], scratch);
                }
            }
        }
        tmp_ct.trace_inplace(module, 0, log_n, auto_keys, scratch);
        tmp_ct
    }

    fn read_prepare_write(
        &mut self,
        params: &Parameters,
        address: &Address,
        auto_keys: &HashMap<i64, AutomorphismKey<Vec<u8>, FFT64>>,
        scratch: &mut Scratch,
    ) -> GLWECiphertext<Vec<u8>> {
        assert_eq!(
            self.state, false,
            "invalid call to Memory.read: internal state is true -> requires calling Memory.write_after_read"
        );

        let module: &Module<FFT64> = &params.module();
        let log_n: usize = module.log_n();
        let basek: usize = params.basek();
        let k_ct: usize = params.k_ct();
        let rank: usize = params.rank();
        let packer: &mut StreamPacker = &mut self.packer;

        for i in 0..address.n2() {
            let coordinate: &Coordinate<Vec<u8>> = address.at(i);

            let res_prev: &mut Vec<GLWECiphertext<Vec<u8>>>;

            if i == 0 {
                res_prev = &mut self.data;
            } else {
                res_prev = &mut self.tree[i - 1];
            }

            // Shift polynomial of the last iteration by X^{-i}
            res_prev.iter_mut().for_each(|poly| {
                coordinate.product_inplace(module, poly, scratch);
            });

            if i < address.n2() - 1 {
                let mut result_next: Vec<GLWECiphertext<Vec<u8>>> = Vec::new();

                // Packs the first coefficient of each polynomial.
                for chunk in res_prev.chunks(module.n()) {
                    for i in 0..module.n() {
                        let i_rev: usize = reverse_bits_msb(i, log_n as u32);
                        if i_rev < chunk.len() {
                            packer.add(
                                module,
                                &mut result_next,
                                Some(&chunk[i_rev]),
                                auto_keys,
                                scratch,
                            );
                        } else {
                            packer.add(
                                module,
                                &mut result_next,
                                None::<&GLWECiphertext<Vec<u8>>>,
                                auto_keys,
                                scratch,
                            );
                        }
                    }
                }

                packer.flush(module, &mut result_next, auto_keys, scratch);
                packer.reset();

                // Stores the packed polynomial
                izip!(self.tree[i].iter_mut(), result_next.iter()).for_each(|(a, b)| {
                    a.copy(module, b);
                });
            }
        }

        let mut res: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(module, basek, k_ct, rank);

        self.state = true;
        if address.n2() != 1 {
            res.copy(module, &self.tree.last().unwrap()[0]);
        } else {
            res.copy(module, &self.data[0]);
        }

        res.trace_inplace(module, 0, log_n, auto_keys, scratch);
        res
    }

    fn write_first_step<DataW: AsRef<[u8]>>(
        &mut self,
        params: &Parameters,
        w: &GLWECiphertext<DataW>,
        n2: usize,
        auto_keys: &HashMap<i64, AutomorphismKey<Vec<u8>, FFT64>>,
        scratch: &mut Scratch,
    ) {
        assert_eq!(
            self.state, true,
            "invalid call to Memory.read: internal state is true -> requires calling Memory.write_after_read"
        );

        let module: &Module<FFT64> = params.module();
        let log_n: usize = module.log_n();
        let basek: usize = params.basek();
        let k_ct: usize = params.k_ct();
        let rank: usize = params.rank();

        let to_write_on: &mut GLWECiphertext<Vec<u8>>;

        if n2 != 1 {
            to_write_on = &mut self.tree.last_mut().unwrap()[0]
        } else {
            to_write_on = &mut self.data[0];
        }

        let (mut tmp_a, scratch_1) = scratch.tmp_glwe_ct(module, basek, k_ct, rank);
        tmp_a.trace::<Vec<u8>, _>(module, 0, log_n, to_write_on, auto_keys, scratch_1);
        to_write_on.sub_inplace_ab(module, &tmp_a);
        to_write_on.add_inplace(module, w);
        to_write_on.normalize_inplace(module, scratch);
    }

    fn write_mid_step<DataCoordinate: AsRef<[u8]>>(
        &mut self,
        step: usize,
        params: &Parameters,
        inv_coordinate: &Coordinate<DataCoordinate>,
        auto_keys: &HashMap<i64, AutomorphismKey<Vec<u8>, FFT64>>,
        scratch: &mut Scratch,
    ) {
        let module: &Module<FFT64> = params.module();
        let log_n: usize = module.log_n();
        let basek: usize = params.basek();
        let k_ct: usize = params.k_ct();
        let rank: usize = params.rank();

        let tree_hi: &mut Vec<GLWECiphertext<Vec<u8>>>; // Above level
        let tree_lo: &mut Vec<GLWECiphertext<Vec<u8>>>; // Current level

        // Top of the tree is not stored in results.
        if step == 0 {
            tree_hi = &mut self.data;
            tree_lo = &mut self.tree[0];
        } else {
            let (left, right) = self.tree.split_at_mut(step);
            tree_hi = &mut left[left.len() - 1];
            tree_lo = &mut right[0];
        }

        tree_hi
            .chunks_mut(module.n())
            .enumerate()
            .for_each(|(j, chunk)| {
                // Retrieve the associated polynomial to extract and pack related to the current chunk
                let ct_lo: &mut GLWECiphertext<Vec<u8>> = &mut tree_lo[j];

                inv_coordinate.product_inplace(module, ct_lo, scratch);

                chunk.iter_mut().for_each(|ct_hi| {
                    // Zeroes the first coefficient of ct_hi
                    // ct_hi = [a, b, c, d] - TRACE([a, b, c, d]) = [0, b, c, d]
                    let (mut tmp_a, scratch_1) = scratch.tmp_glwe_ct(module, basek, k_ct, rank);
                    tmp_a.trace::<Vec<u8>, Vec<u8>>(module, 0, log_n, ct_hi, auto_keys, scratch_1);
                    ct_hi.sub_inplace_ab(module, &tmp_a);

                    // Extract the first coefficient ct_lo
                    // tmp_a = TRACE([a, b, c, d]) -> [a, 0, 0, 0]
                    tmp_a.trace::<Vec<u8>, Vec<u8>>(module, 0, log_n, ct_lo, auto_keys, scratch_1);

                    // Adds extracted coefficient of ct_lo on ct_hi
                    // [a, 0, 0, 0] + [0, b, c, d]
                    ct_hi.add_inplace(module, &tmp_a);
                    ct_hi.normalize_inplace(module, scratch_1);

                    // Cyclic shift ct_lo by X^-1
                    ct_lo.rotate_inplace(module, -1);
                })
            });
    }

    fn write_last_step<DataCoordinate: AsRef<[u8]>>(
        &mut self,
        module: &Module<FFT64>,
        inv_coordinate: &Coordinate<DataCoordinate>,
        scratch: &mut Scratch,
    ) {
        // Apply the last reverse shift to the top of the tree.
        self.data.iter_mut().for_each(|ct_lo| {
            inv_coordinate.product_inplace(module, ct_lo, scratch);
        });
        self.state = false;
    }
}
