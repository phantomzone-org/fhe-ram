use std::collections::HashMap;

use poulpy_backend::FFT64Spqlios;
use poulpy_core::{
    GLWEOperations, GLWEPacker, TakeGLWECt,
    layouts::{
        GGLWEAutomorphismKey, GGLWETensorKey, GLWECiphertext, GLWEPlaintext, GLWESecret,
        prepared::{
            GGLWEAutomorphismKeyPrepared, GGLWETensorKeyPrepared, GLWESecretPrepared, PrepareAlloc,
        },
    },
};
use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Data, Module, Scratch, ScratchOwned},
    source::Source,
};

use itertools::izip;
use rand_core::{OsRng, TryRngCore};

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
    pub(crate) scratch: ScratchOwned<FFT64Spqlios>,
}

impl Ram {
    /// Instantiates a new [Ram].
    pub fn new() -> Self {
        let params: Parameters = Parameters::default();
        let scratch: ScratchOwned<FFT64Spqlios> = ScratchOwned::alloc(Self::scratch_bytes(&params));
        Self {
            subrams: (0..params.word_size())
                .map(|_| SubRam::alloc(&params))
                .collect(),
            params,
            scratch,
        }
    }

    /// Scratch space size required by the [Ram].
    pub(crate) fn scratch_bytes(params: &Parameters) -> usize {
        let module: &Module<FFT64Spqlios> = params.module();
        let k_ct: usize = params.k_ct();
        let k_evk: usize = params.k_evk();
        let basek: usize = params.basek();
        let rank: usize = params.rank();
        let digits: usize = params.digits();

        let enc_sk: usize = GLWECiphertext::encrypt_sk_scratch_space(module, basek, k_ct);
        let coordinate_product: usize = Coordinate::product_scratch_space(params);
        let packing: usize = GLWEPacker::scratch_space(module, basek, k_ct, k_evk, digits, rank);
        let trace: usize =
            GLWECiphertext::trace_inplace_scratch_space(module, basek, k_ct, k_evk, digits, rank);
        let ct: usize = GLWECiphertext::bytes_of(module.n(), basek, k_ct, rank);
        let inv_addr: usize = Coordinate::invert_scratch_space(params);

        let read: usize = coordinate_product | trace | packing;
        let write: usize = coordinate_product | (ct + trace) | inv_addr;

        enc_sk | read | write
    }

    /// Initialize the FHE-[Ram] with the provided values (encrypted under the provided secret).
    pub fn encrypt_sk(&mut self, data: &[u8], sk: &GLWESecret<Vec<u8>>) {
        let params: &Parameters = &self.params;
        let max_addr: usize = params.max_addr();
        let ram_chunks: usize = params.word_size();

        assert!(
            data.len().is_multiple_of(ram_chunks),
            "invalid data: data.len()%ram_chunks={} != 0",
            data.len() % ram_chunks,
        );

        assert!(
            data.len() / ram_chunks == max_addr,
            "invalid data: data.len()/ram_chunks={} != max_addr={}",
            data.len() / ram_chunks,
            max_addr
        );

        let scratch: &mut Scratch<FFT64Spqlios> = self.scratch.borrow();

        let mut data_split: Vec<u8> = vec![0u8; max_addr];
        (0..ram_chunks).for_each(|i| {
            data_split.iter_mut().enumerate().for_each(|(j, x)| {
                *x = data[j * ram_chunks + i];
            });
            self.subrams[i].encrypt_sk(params, &data_split, sk, scratch);
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
            !self.subrams.is_empty(),
            "unitialized memory: self.data.len()=0"
        );

        self.subrams
            .iter_mut()
            .map(|subram| {
                subram.read(
                    &self.params,
                    address,
                    &keys.auto_keys,
                    self.scratch.borrow(),
                )
            })
            .collect()
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
            !self.subrams.is_empty(),
            "unitialized memory: self.data.len()=0"
        );

        self.subrams
            .iter_mut()
            .map(|subram| {
                subram.read_prepare_write(
                    &self.params,
                    address,
                    &keys.auto_keys,
                    self.scratch.borrow(),
                )
            })
            .collect()
    }

    /// Writes w to the [Ram]. Requires that [Self::read_prepare_write] was
    /// called beforehand.
    pub fn write<DataW: Data + AsRef<[u8]>>(
        &mut self,
        w: &[GLWECiphertext<DataW>], // Must encrypt [w, 0, 0, ..., 0];
        address: &Address,
        keys: &EvaluationKeys,
    ) {
        assert!(w.len() == self.subrams.len());

        let params: &Parameters = &self.params;
        let module: &Module<FFT64Spqlios> = params.module();
        let basek: usize = params.basek();
        let rank: usize = params.rank();
        let digits: usize = params.digits();

        let scratch: &mut Scratch<FFT64Spqlios> = self.scratch.borrow();
        let auto_keys: &HashMap<i64, GGLWEAutomorphismKey<Vec<u8>>> = &keys.auto_keys;
        let tensor_key: &GGLWETensorKey<Vec<u8>> = &keys.tensor_key;

        // Overwrites the coefficient that was read: to_write_on = to_write_on - TRACE(to_write_on) + w
        self.subrams.iter_mut().enumerate().for_each(|(i, subram)| {
            subram.write_first_step(params, &w[i], address.n2(), auto_keys, scratch);
        });

        let auto_key_prepared: GGLWEAutomorphismKeyPrepared<Vec<u8>, FFT64Spqlios> =
            auto_keys.get(&-1).unwrap().prepare_alloc(module, scratch);

        let tensor_key_prepared: GGLWETensorKeyPrepared<Vec<u8>, FFT64Spqlios> =
            tensor_key.prepare_alloc(module, scratch);

        for i in (0..address.n2() - 1).rev() {
            // Index polynomial X^{i}
            let coordinate: &Coordinate<Vec<u8>> = address.at(i + 1);

            let mut inv_coordinate: Coordinate<Vec<u8>> = Coordinate::alloc(
                module,
                basek,
                address.k(),
                address.rows(),
                rank,
                digits,
                &coordinate.base1d,
            ); // DODO ALLOC FROM SCRATCH SPACE

            inv_coordinate.base1d = coordinate.base1d.clone();

            // Inverts coordinate: X^{i} -> X^{-i}
            inv_coordinate.invert(
                module,
                coordinate,
                &auto_key_prepared,
                &tensor_key_prepared,
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
            digits,
            &coordinate.base1d.clone(),
        ); // DODO ALLOC FROM SCRATCH SPACE

        // Inverts coordinate: X^{i} -> X^{-i}
        inv_coordinate.invert(
            module,
            coordinate,
            &auto_key_prepared,
            &tensor_key_prepared,
            scratch,
        );

        self.subrams.iter_mut().for_each(|subram| {
            subram.write_last_step(module, &inv_coordinate, scratch);
        })
    }
}

impl Default for Ram {
    fn default() -> Self {
        Self::new()
    }
}

/// [SubRam] stores a digit of the word.
pub(crate) struct SubRam {
    data: Vec<GLWECiphertext<Vec<u8>>>,
    tree: Vec<Vec<GLWECiphertext<Vec<u8>>>>,
    packer: GLWEPacker,
    state: bool,
}

impl SubRam {
    pub fn alloc(params: &Parameters) -> Self {
        let module: &Module<FFT64Spqlios> = params.module();
        let basek: usize = params.basek();
        let k_ct: usize = params.k_ct();
        let rank: usize = params.rank();

        let n: usize = module.n();
        let mut tree: Vec<Vec<GLWECiphertext<Vec<u8>>>> = Vec::new();
        let max_addr_split: usize = params.max_addr(); // u8 -> u32

        if max_addr_split > n {
            let mut size: usize = max_addr_split.div_ceil(n);
            while size != 1 {
                size = size.div_ceil(n);
                let tmp: Vec<GLWECiphertext<Vec<u8>>> = (0..size)
                    .map(|_| GLWECiphertext::alloc(module.n(), basek, k_ct, rank))
                    .collect();
                tree.push(tmp);
            }
        }

        Self {
            data: Vec::new(),
            tree,
            packer: GLWEPacker::new(module.n(), 0, basek, k_ct, rank),
            state: false,
        }
    }

    pub fn encrypt_sk(
        &mut self,
        params: &Parameters,
        data: &[u8],
        sk: &GLWESecret<Vec<u8>>,
        scratch: &mut Scratch<FFT64Spqlios>,
    ) {
        let module: &Module<FFT64Spqlios> = params.module();
        let k_pt: usize = params.k_pt();
        let rank: usize = params.rank();
        let basek: usize = params.basek();
        let k_ct: usize = params.k_ct();

        let mut root = [0u8; 32];
        OsRng.try_fill_bytes(&mut root).unwrap();

        let mut source: Source = Source::new(root);

        let seed_xa = source.new_seed();
        let mut source_xa = Source::new(seed_xa);

        let seed_xe = source.new_seed();
        let mut source_xe = Source::new(seed_xe);

        let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(module.n(), basek, k_pt);
        let mut data_i64: Vec<i64> = vec![0i64; module.n()];

        self.data = data
            .chunks(module.n())
            .map(|chunk| {
                let mut ct: GLWECiphertext<Vec<u8>> =
                    GLWECiphertext::alloc(module.n(), basek, k_ct, rank);
                izip!(data_i64.iter_mut(), chunk.iter())
                    .for_each(|(xi64, xu8)| *xi64 = *xu8 as i64);
                data_i64[chunk.len()..].iter_mut().for_each(|x| *x = 0);
                pt.data.encode_vec_i64(basek, 0, k_pt, &data_i64, 8);
                let sk_prepared: GLWESecretPrepared<Vec<u8>, FFT64Spqlios> =
                    sk.prepare_alloc(module, scratch);

                ct.encrypt_sk(
                    module,
                    &pt,
                    &sk_prepared,
                    &mut source_xa,
                    &mut source_xe,
                    scratch,
                );
                ct
            })
            .collect();
    }

    fn read(
        &mut self,
        params: &Parameters,
        address: &Address,
        auto_keys: &HashMap<i64, GGLWEAutomorphismKey<Vec<u8>>>,
        scratch: &mut Scratch<FFT64Spqlios>,
    ) -> GLWECiphertext<Vec<u8>> {
        assert!(
            !self.state,
            "invalid call to Memory.read: internal state is true -> requires calling Memory.write"
        );

        let module: &Module<FFT64Spqlios> = params.module();
        let log_n: usize = module.log_n();
        let basek: usize = params.basek();
        let k_ct: usize = params.k_ct();
        let rank: usize = params.rank();
        let packer: &mut GLWEPacker = &mut self.packer;

        let mut results: Vec<GLWECiphertext<Vec<u8>>> = Vec::new();
        let mut tmp_ct: GLWECiphertext<Vec<u8>> =
            GLWECiphertext::alloc(module.n(), basek, k_ct, rank);
        let auto_keys_prepared: HashMap<i64, GGLWEAutomorphismKeyPrepared<Vec<u8>, FFT64Spqlios>> =
            auto_keys
                .iter()
                .map(|(g, k)| (*g, k.prepare_alloc(module, scratch)))
                .collect();

        for i in 0..address.n2() {
            let coordinate: &Coordinate<Vec<u8>> = address.at(i);
            let res_prev: &Vec<GLWECiphertext<Vec<u8>>> =
                if i == 0 { &self.data } else { &results };

            if i < address.n2() - 1 {
                let mut result_next: Vec<GLWECiphertext<Vec<u8>>> = Vec::new();

                for chunk in res_prev.chunks(module.n()) {
                    (0..module.n()).for_each(|j| {
                        let j_rev = reverse_bits_msb(j, log_n as u32);

                        if j_rev < chunk.len() {
                            coordinate.product(module, &mut tmp_ct, &chunk[j_rev], scratch);
                            packer.add(module, Some(&tmp_ct), &auto_keys_prepared, scratch);
                        } else {
                            packer.add(
                                module,
                                None::<&GLWECiphertext<Vec<u8>>>,
                                &auto_keys_prepared,
                                scratch,
                            );
                        }
                    });
                    let mut packed_ct = GLWECiphertext::alloc(module.n(), basek, k_ct, rank);
                    packer.flush(module, &mut packed_ct);
                    result_next.push(packed_ct);
                }

                results = result_next;
            } else if i == 0 {
                coordinate.product(module, &mut tmp_ct, &self.data[0], scratch);
            } else {
                coordinate.product(module, &mut tmp_ct, &results[0], scratch);
            }
        }
        tmp_ct.trace_inplace(module, 0, log_n, &auto_keys_prepared, scratch);
        tmp_ct
    }

    fn read_prepare_write(
        &mut self,
        params: &Parameters,
        address: &Address,
        auto_keys: &HashMap<i64, GGLWEAutomorphismKey<Vec<u8>>>,
        scratch: &mut Scratch<FFT64Spqlios>,
    ) -> GLWECiphertext<Vec<u8>> {
        assert!(
            !self.state,
            "invalid call to Memory.read: internal state is true -> requires calling Memory.write"
        );

        let module: &Module<FFT64Spqlios> = params.module();
        let log_n: usize = module.log_n();
        let basek: usize = params.basek();
        let k_ct: usize = params.k_ct();
        let rank: usize = params.rank();
        let packer: &mut GLWEPacker = &mut self.packer;
        let auto_keys_prepared: HashMap<i64, GGLWEAutomorphismKeyPrepared<Vec<u8>, FFT64Spqlios>> =
            auto_keys
                .iter()
                .map(|(g, k)| (*g, k.prepare_alloc(module, scratch)))
                .collect();

        for i in 0..address.n2() {
            let coordinate: &Coordinate<Vec<u8>> = address.at(i);

            let res_prev: &mut Vec<GLWECiphertext<Vec<u8>>> = if i == 0 {
                &mut self.data
            } else {
                &mut self.tree[i - 1]
            };

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
                            packer.add(module, Some(&chunk[i_rev]), &auto_keys_prepared, scratch);
                        } else {
                            packer.add(
                                module,
                                None::<&GLWECiphertext<Vec<u8>>>,
                                &auto_keys_prepared,
                                scratch,
                            );
                        }
                    }
                    let mut packed_ct = GLWECiphertext::alloc(module.n(), basek, k_ct, rank);
                    packer.flush(module, &mut packed_ct);
                    result_next.push(packed_ct);
                }

                // Stores the packed polynomial
                izip!(self.tree[i].iter_mut(), result_next.iter()).for_each(|(a, b)| {
                    a.copy(module, b);
                });
            }
        }

        let mut res: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(module.n(), basek, k_ct, rank);

        self.state = true;
        if address.n2() != 1 {
            res.copy(module, &self.tree.last().unwrap()[0]);
        } else {
            res.copy(module, &self.data[0]);
        }

        res.trace_inplace(module, 0, log_n, &auto_keys_prepared, scratch);
        res
    }

    fn write_first_step<DataW: Data + AsRef<[u8]>>(
        &mut self,
        params: &Parameters,
        w: &GLWECiphertext<DataW>,
        n2: usize,
        auto_keys: &HashMap<i64, GGLWEAutomorphismKey<Vec<u8>>>,
        scratch: &mut Scratch<FFT64Spqlios>,
    ) {
        assert!(
            self.state,
            "invalid call to Memory.write: internal state is false -> requires calling Memory.read_prepare_write"
        );

        let module: &Module<FFT64Spqlios> = params.module();
        let log_n: usize = module.log_n();
        let basek: usize = params.basek();
        let k_ct: usize = params.k_ct();
        let rank: usize = params.rank();

        let auto_keys_prepared: HashMap<i64, GGLWEAutomorphismKeyPrepared<Vec<u8>, FFT64Spqlios>> =
            auto_keys
                .iter()
                .map(|(g, k)| (*g, k.prepare_alloc(module, scratch)))
                .collect();

        let to_write_on: &mut GLWECiphertext<Vec<u8>> = if n2 != 1 {
            &mut self.tree.last_mut().unwrap()[0]
        } else {
            &mut self.data[0]
        };

        let (mut tmp_a, scratch_1) = scratch.take_glwe_ct(module.n(), basek, k_ct, rank);
        tmp_a.trace::<Vec<u8>, _, _>(
            module,
            0,
            log_n,
            to_write_on,
            &auto_keys_prepared,
            scratch_1,
        );
        to_write_on.sub_inplace_ab(module, &tmp_a);
        to_write_on.add_inplace(module, w);
        to_write_on.normalize_inplace(module, scratch);
    }

    fn write_mid_step<DataCoordinate: Data + AsRef<[u8]>>(
        &mut self,
        step: usize,
        params: &Parameters,
        inv_coordinate: &Coordinate<DataCoordinate>,
        auto_keys: &HashMap<i64, GGLWEAutomorphismKey<Vec<u8>>>,
        scratch: &mut Scratch<FFT64Spqlios>,
    ) {
        let module: &Module<FFT64Spqlios> = params.module();
        let log_n: usize = module.log_n();
        let basek: usize = params.basek();
        let k_ct: usize = params.k_ct();
        let rank: usize = params.rank();

        let tree_hi: &mut Vec<GLWECiphertext<Vec<u8>>>; // Above level
        let tree_lo: &mut Vec<GLWECiphertext<Vec<u8>>>; // Current level

        let auto_keys_prepared: HashMap<i64, GGLWEAutomorphismKeyPrepared<Vec<u8>, FFT64Spqlios>> =
            auto_keys
                .iter()
                .map(|(g, k)| (*g, k.prepare_alloc(module, scratch)))
                .collect();

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
                    let (mut tmp_a, scratch_1) =
                        scratch.take_glwe_ct(module.n(), basek, k_ct, rank);
                    tmp_a.trace::<Vec<u8>, Vec<u8>, FFT64Spqlios>(
                        module,
                        0,
                        log_n,
                        ct_hi,
                        &auto_keys_prepared,
                        scratch_1,
                    );
                    ct_hi.sub_inplace_ab(module, &tmp_a);

                    // Extract the first coefficient ct_lo
                    // tmp_a = TRACE([a, b, c, d]) -> [a, 0, 0, 0]
                    tmp_a.trace::<Vec<u8>, Vec<u8>, FFT64Spqlios>(
                        module,
                        0,
                        log_n,
                        ct_lo,
                        &auto_keys_prepared,
                        scratch_1,
                    );

                    // Adds extracted coefficient of ct_lo on ct_hi
                    // [a, 0, 0, 0] + [0, b, c, d]
                    ct_hi.add_inplace(module, &tmp_a);
                    ct_hi.normalize_inplace(module, scratch_1);

                    // Cyclic shift ct_lo by X^-1
                    ct_lo.rotate_inplace(module, -1, scratch_1);
                })
            });
    }

    fn write_last_step<DataCoordinate: Data + AsRef<[u8]>>(
        &mut self,
        module: &Module<FFT64Spqlios>,
        inv_coordinate: &Coordinate<DataCoordinate>,
        scratch: &mut Scratch<FFT64Spqlios>,
    ) {
        // Apply the last reverse shift to the top of the tree.
        self.data.iter_mut().for_each(|ct_lo| {
            inv_coordinate.product_inplace(module, ct_lo, scratch);
        });
        self.state = false;
    }
}
