use core::{
    automorphism::AutomorphismKey,
    elem::Infos,
    glwe_ciphertext::GLWECiphertext,
    glwe_plaintext::GLWEPlaintext,
    keys::{SecretKey, SecretKeyFourier},
    tensor_key::TensorKey,
};
use std::collections::HashMap;

use backend::{Encoding, FFT64, Module, Scratch, ScratchOwned, VecZnx, VecZnxAlloc, VecZnxToRef};
use itertools::izip;
use sampling::source::{Source, new_seed};

use crate::{
    address::{Address, Coordinate},
    keys::EvaluationKeys,
    packing::StreamPacker,
    parameters::Parameters,
    reverse_bits_msb,
};

pub struct Ram {
    pub(crate) params: Parameters,
    pub(crate) data: Vec<GLWECiphertext<Vec<u8>>>,
    pub(crate) tree: Vec<Vec<GLWECiphertext<Vec<u8>>>>,
    pub(crate) state: bool,
    pub(crate) scratch: ScratchOwned,
}

impl Ram {
    pub fn new() -> Self {
        let params: Parameters = Parameters::new();
        let module: &Module<FFT64> = &params.module();
        let basek: usize = params.basek();
        let k_ct: usize = params.k_ct();
        let rank: usize = params.rank();

        let n: usize = module.n();
        let mut tree: Vec<Vec<GLWECiphertext<Vec<u8>>>> = Vec::new();
        let max_addr: usize = params.max_addr();

        if max_addr > n {
            let mut size: usize = (max_addr + n - 1) / n;

            while size != 1 {
                size = (size + n - 1) / n;
                let mut tmp: Vec<GLWECiphertext<Vec<u8>>> = Vec::new();
                (0..size).for_each(|_| tmp.push(GLWECiphertext::alloc(module, basek, k_ct, rank)));
                tree.push(tmp);
            }
        }

        let scratch: ScratchOwned = ScratchOwned::new(Self::scratch_bytes(&params));

        Self {
            data: Vec::new(),
            params: params,
            tree: tree,
            state: false,
            scratch,
        }
    }

    pub(crate) fn scratch_bytes(params: &Parameters) -> usize {
        let module: &Module<FFT64> = params.module();
        let k_ct: usize = params.k_ct();
        let k_evk: usize = params.k_evk();
        let basek: usize = params.basek();
        let rank: usize = params.rank();

        let ct_glwe_size: usize = (k_ct + basek - 1) / basek;
        let autokey_size: usize = (k_evk + basek - 1) / basek;

        let enc_sk: usize = GLWECiphertext::encrypt_sk_scratch_space(module, ct_glwe_size);
        let coordinate_product: usize = Coordinate::product_scratch_space(params);
        let packing: usize =
            StreamPacker::add_scratch_space(module, ct_glwe_size, autokey_size, rank);
        let trace: usize =
            GLWECiphertext::trace_inplace_scratch_space(module, ct_glwe_size, autokey_size, rank);
        let vec_znx: usize = module.bytes_of_vec_znx(rank + 1, ct_glwe_size);
        let inv_addr: usize = Coordinate::invert_scratch_space(params);

        let read: usize = coordinate_product | trace | packing;
        let write: usize = coordinate_product | 2 * vec_znx + trace | inv_addr;

        enc_sk | read | write
    }

    pub fn encrypt_sk(&mut self, data: &[u8], sk: &SecretKey<Vec<u8>>) {
        let max_addr: usize = self.params.max_addr();

        assert!(
            data.len() <= max_addr,
            "invalid data: data.len()={} > max_addr={}",
            data.len(),
            max_addr
        );

        let params: &Parameters = &self.params;
        let module: &Module<FFT64> = &params.module();
        let k_pt: usize = params.k_pt();
        let sigma: f64 = params.xe();
        let scratch: &mut Scratch = self.scratch.borrow();
        let rank: usize = params.rank();
        let basek: usize = params.basek();
        let k_ct: usize = params.k_ct();

        let mut sk_dft: SecretKeyFourier<Vec<u8>, FFT64> = SecretKeyFourier::alloc(module, rank);
        sk_dft.dft(module, sk);

        let mut source_xa: Source = Source::new(new_seed());
        let mut source_xe: Source = Source::new(new_seed());

        let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(module, basek, k_pt);
        let mut cts: Vec<GLWECiphertext<Vec<u8>>> = Vec::new();

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

        self.data = cts;
    }

    pub fn read(&mut self, address: &Address, keys: &EvaluationKeys) -> GLWECiphertext<Vec<u8>> {
        assert!(
            self.data.len() != 0,
            "unitialized memory: self.data.len()=0"
        );
        assert_eq!(
            self.state, false,
            "invalid call to Memory.read: internal state is true -> requires calling Memory.write"
        );

        let params: &Parameters = &self.params;
        let module: &Module<FFT64> = &params.module();
        let basek: usize = params.basek();
        let k_ct: usize = params.k_ct();
        let rank: usize = params.rank();
        let scratch: &mut Scratch = self.scratch.borrow();
        let auto_keys: &HashMap<i64, AutomorphismKey<Vec<u8>, FFT64>> = &keys.auto_keys;

        let log_n: usize = module.log_n();

        let mut packer: StreamPacker = StreamPacker::new(module, basek, k_ct, rank);
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
                    for j in 0..module.n() {
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
                    }
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

        tmp_ct.trace_inplace(module, 0, module.log_n(), auto_keys, scratch);
        tmp_ct
    }

    pub fn read_prepare_write(
        &mut self,
        address: &Address,
        keys: &EvaluationKeys,
    ) -> GLWECiphertext<Vec<u8>> {
        assert!(
            self.data.len() != 0,
            "unitialized memory: self.data.len()=0"
        );
        assert_eq!(
            self.state, false,
            "invalid call to Memory.read: internal state is true -> requires calling Memory.write_after_read"
        );

        let params: &Parameters = &self.params;
        let module: &Module<FFT64> = &params.module();
        let log_n: usize = module.log_n();
        let basek: usize = params.basek();
        let k_ct: usize = params.k_ct();
        let rank: usize = params.rank();
        let scratch: &mut Scratch = self.scratch.borrow();
        let auto_keys: &HashMap<i64, AutomorphismKey<Vec<u8>, FFT64>> = &keys.auto_keys;

        let mut packer: StreamPacker = StreamPacker::new(module, basek, k_ct, rank);

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
            res.copy(module, &self.tree.last().unwrap()[0])
        } else {
            res.copy(module, &self.data[0]);
        }

        res.trace_inplace(module, 0, log_n, auto_keys, scratch);
        res
    }

    pub fn write<DataW>(
        &mut self,
        w: &GLWECiphertext<DataW>, // Must encrypt [w, 0, 0, ..., 0];
        address: &Address,
        keys: &EvaluationKeys,
        sk: &SecretKey<Vec<u8>>,
    ) where
        VecZnx<DataW>: VecZnxToRef,
        DataW: std::convert::AsRef<[u8]>, // TODO FIX THIS
    {
        assert_eq!(
            self.state, true,
            "invalid call to Memory.read: internal state is true -> requires calling Memory.write_after_read"
        );

        let params: &Parameters = &self.params;
        let module: &Module<FFT64> = &params.module();
        let log_n: usize = module.log_n();
        let basek: usize = params.basek();
        let k_ct: usize = params.k_ct();
        let rank: usize = params.rank();
        let size: usize = self.data[0].size();
        let scratch: &mut Scratch = self.scratch.borrow();
        let auto_keys: &HashMap<i64, AutomorphismKey<_, FFT64>> = &keys.auto_keys;
        let tensor_key: &TensorKey<_, FFT64> = &keys.tensor_key;

        let mut sk_dft: SecretKeyFourier<Vec<u8>, FFT64> = SecretKeyFourier::alloc(module, rank);
        sk_dft.dft(module, sk);

        // Overwrites the coefficient that was read: to_write_on = to_write_on - TRACE(to_write_on) + w
        let to_write_on: &mut GLWECiphertext<Vec<u8>>;

        if address.n2() != 1 {
            to_write_on = &mut self.tree.last_mut().unwrap()[0];
        } else {
            to_write_on = &mut self.data[0]
        }

        let (tmp_a_data, scratch_1) = scratch.tmp_vec_znx(module, rank + 1, size);
        let mut tmp_a: GLWECiphertext<&mut [u8]> = GLWECiphertext {
            data: tmp_a_data,
            k: k_ct,
            basek: basek,
        };
        tmp_a.trace::<Vec<u8>, _>(module, 0, log_n, to_write_on, auto_keys, scratch_1);
        to_write_on.sub_inplace_ab(module, &tmp_a);
        to_write_on.add_inplace(module, w);
        to_write_on.normalize_inplace(module, scratch);

        for i in (0..address.n2() - 1).rev() {
            // Index polynomial X^{i}
            let coordinate: &Coordinate<Vec<u8>> = address.at(i + 1);

            let mut coordinate_inv: Coordinate<Vec<u8>> = Coordinate::alloc(
                module,
                basek,
                address.k(),
                address.rows(),
                rank,
                &coordinate.base1d.clone(),
            ); // DODO ALLOC FROM SCRATCH SPACE

            coordinate_inv.base1d = coordinate.base1d.clone();

            // Inverts coordinate: X^{i} -> X^{-i}
            coordinate_inv.invert(
                module,
                coordinate,
                auto_keys.get(&-1).unwrap(),
                tensor_key,
                scratch,
                sk,
            );

            let result_hi: &mut Vec<GLWECiphertext<Vec<u8>>>; // Above level
            let result_lo: &mut Vec<GLWECiphertext<Vec<u8>>>; // Current level

            // Top of the tree is not stored in results.
            if i == 0 {
                result_hi = &mut self.data;
                result_lo = &mut self.tree[0];
            } else {
                let (left, right) = self.tree.split_at_mut(i);
                result_hi = &mut left[left.len() - 1];
                result_lo = &mut right[0];
            }

            result_hi
                .chunks_mut(module.n())
                .enumerate()
                .for_each(|(j, chunk)| {
                    // Retrieve the associated polynomial to extract and pack related to the current chunk
                    let ct_lo: &mut GLWECiphertext<Vec<u8>> = &mut result_lo[j];

                    coordinate_inv.product_inplace(module, ct_lo, scratch);

                    chunk.iter_mut().for_each(|ct_hi| {
                        // Extract the first coefficient ct_lo
                        // tmp_a = TRACE([a, b, c, d]) -> [a, 0, 0, 0]
                        let (tmp_a_data, scratch_1) = scratch.tmp_vec_znx(module, rank + 1, size);
                        let mut tmp_a: GLWECiphertext<&mut [u8]> = GLWECiphertext {
                            data: tmp_a_data,
                            k: k_ct,
                            basek: basek,
                        };
                        tmp_a.trace::<Vec<u8>, _>(module, 0, log_n, ct_lo, auto_keys, scratch_1);

                        // Zeroes the first coefficient of ct_hi
                        // ct_hi = [a, b, c, d] - TRACE([a, b, c, d]) = [0, b, c, d]
                        let (tmp_b_data, scratch_2) = scratch_1.tmp_vec_znx(module, rank + 1, size);
                        let mut tmp_b: GLWECiphertext<&mut [u8]> = GLWECiphertext {
                            data: tmp_b_data,
                            k: k_ct,
                            basek: basek,
                        };
                        tmp_b.trace::<Vec<u8>, _>(module, 0, log_n, ct_hi, auto_keys, scratch_2);

                        ct_hi.sub_inplace_ab(module, &tmp_b);

                        // Adds extracted coefficient of ct_lo on ct_hi
                        // [a, 0, 0, 0] + [0, b, c, d]
                        ct_hi.add_inplace(module, &tmp_a);
                        ct_hi.normalize_inplace(module, scratch_2);

                        // Cyclic shift ct_lo by X^-1
                        ct_lo.rotate_inplace(module, -1);
                    })
                });
        }

        // Apply the last reverse shift to the top of the tree.
        self.data.iter_mut().for_each(|ct_lo| {
            let coordinate: &Coordinate<Vec<u8>> = address.at(0);

            let mut coordinate_inv: Coordinate<Vec<u8>> = Coordinate::alloc(
                module,
                basek,
                address.k(),
                address.rows(),
                rank,
                &coordinate.base1d.clone(),
            ); // DODO ALLOC FROM SCRATCH SPACE

            // Inverts coordinate: X^{i} -> X^{-i}
            coordinate_inv.invert(
                module,
                coordinate,
                auto_keys.get(&-1).unwrap(),
                tensor_key,
                scratch,
                sk,
            );

            coordinate_inv.product_inplace(module, ct_lo, scratch);
        });

        self.state = false;
    }
}
