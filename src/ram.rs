use core::{
    automorphism::AutomorphismKey, elem::Infos, glwe_ciphertext::GLWECiphertext,
    glwe_plaintext::GLWEPlaintext, keys::SecretKeyFourier, tensor_key::TensorKey,
};
use std::collections::HashMap;

use backend::{
    Encoding, FFT64, MatZnxDft, MatZnxDftToRef, Module, ScalarZnxDft, ScalarZnxDftToRef, Scratch,
    VecZnx, VecZnxToRef,
};
use itertools::izip;
use sampling::source::Source;

use crate::{
    address::{Address, Coordinate},
    packing::StreamPacker,
    reverse_bits_msb,
};

pub struct Ram {
    pub(crate) data: Vec<GLWECiphertext<Vec<u8>>>,
    pub(crate) k_ct: usize,
    pub(crate) basek: usize,
    pub(crate) rank: usize,
    pub(crate) max_size: usize,
    pub(crate) k_pt: usize,
    pub(crate) tree: Vec<Vec<GLWECiphertext<Vec<u8>>>>,
    pub(crate) state: bool,
}

impl Ram {
    pub(crate) fn new(
        module: &Module<FFT64>,
        basek: usize,
        k_ct: usize,
        rank: usize,
        max_size: usize,
    ) -> Self {
        let n: usize = module.n();
        let mut tree: Vec<Vec<GLWECiphertext<Vec<u8>>>> = Vec::new();

        if max_size > n {
            let mut size: usize = (max_size + n - 1) / n;

            while size != 1 {
                size = (size + n - 1) / n;
                let mut tmp: Vec<GLWECiphertext<Vec<u8>>> = Vec::new();
                (0..size).for_each(|_| tmp.push(GLWECiphertext::alloc(module, basek, k_ct, rank)));
                tree.push(tmp);
            }
        }

        Self {
            data: Vec::new(),
            basek: basek,
            k_ct: k_ct,
            rank: rank,
            k_pt: 0,
            max_size: max_size,
            tree: tree,
            state: false,
        }
    }

    pub(crate) fn encrypt_sk<DataSk>(
        &mut self,
        module: &Module<FFT64>,
        sk_dft: &SecretKeyFourier<DataSk, FFT64>,
        data: &[i64],
        k_pt: usize,
        source_xa: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        scratch: &mut Scratch,
    ) where
        ScalarZnxDft<DataSk, FFT64>: ScalarZnxDftToRef<FFT64>,
    {
        assert!(
            data.len() <= self.max_size,
            "invalid data: data.len()={} > self.max_size={}",
            data.len(),
            self.max_size
        );

        let basek: usize = self.basek;
        let k_ct: usize = self.k_ct;
        let rank: usize = self.rank;

        let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(module, self.basek, k_pt);

        let mut cts: Vec<GLWECiphertext<Vec<u8>>> = Vec::new();

        for chunk in data.chunks(module.n()) {
            let mut ct: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(module, basek, k_ct, rank);
            pt.data.encode_vec_i64(0, basek, k_pt, chunk, 32);
            ct.encrypt_sk(module, &pt, sk_dft, source_xa, source_xe, sigma, scratch);
            cts.push(ct);
        }

        self.data = cts;
        self.k_pt = k_pt;
    }

    pub(crate) fn read<DataAK>(
        &self,
        module: &Module<FFT64>,
        address: &Address,
        auto_keys: &HashMap<i64, AutomorphismKey<DataAK, FFT64>>,
        scratch: &mut Scratch,
    ) -> GLWECiphertext<Vec<u8>>
    where
        MatZnxDft<DataAK, FFT64>: MatZnxDftToRef<FFT64>,
    {
        assert!(
            self.data.len() != 0,
            "unitialized memory: self.data.len()=0"
        );
        assert_eq!(
            self.state, false,
            "invalid call to Memory.read: internal state is true -> requires calling Memory.write"
        );

        let basek: usize = self.basek;
        let k_ct: usize = self.k_ct;
        let rank: usize = self.rank;

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

    pub(crate) fn read_prepare_write<DataAdr, DataAK>(
        &mut self,
        module: &Module<FFT64>,
        address: &Address,
        auto_keys: &HashMap<i64, AutomorphismKey<DataAK, FFT64>>,
        scratch: &mut Scratch,
    ) -> GLWECiphertext<Vec<u8>>
    where
        MatZnxDft<DataAK, FFT64>: MatZnxDftToRef<FFT64>,
    {
        assert!(
            self.data.len() != 0,
            "unitialized memory: self.data.len()=0"
        );
        assert_eq!(
            self.state, false,
            "invalid call to Memory.read: internal state is true -> requires calling Memory.write_after_read"
        );
        let log_n: usize = module.log_n();
        let basek: usize = self.basek;
        let k_ct: usize = self.k_ct;
        let rank: usize = self.rank;

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

    pub(crate) fn write<DataW, DataAK, DataTK>(
        &mut self,
        module: &Module<FFT64>,
        w: &GLWECiphertext<DataW>, // Must encrypt [w, 0, 0, ..., 0];
        address: &Address,
        auto_keys: &HashMap<i64, AutomorphismKey<DataAK, FFT64>>,
        tensor_key: &TensorKey<DataTK, FFT64>,
        scratch: &mut Scratch,
    ) where
        VecZnx<DataW>: VecZnxToRef,
        MatZnxDft<DataAK, FFT64>: MatZnxDftToRef<FFT64>,
        MatZnxDft<DataTK, FFT64>: MatZnxDftToRef<FFT64>,
    {
        assert_eq!(
            self.state, true,
            "invalid call to Memory.read: internal state is true -> requires calling Memory.write_after_read"
        );

        let log_n: usize = module.log_n();
        let basek: usize = self.basek;
        let k_ct: usize = self.k_ct;
        let rank: usize = self.rank;
        let size: usize = self.data[0].size();

        if address.n2() != 1 {
            let result: &mut GLWECiphertext<Vec<u8>> = &mut self.tree.last_mut().unwrap()[0];
            result.add_inplace(module, w);
            result.normalize_inplace(module, scratch);
        } else {
            self.data[0].add_inplace(module, w);
            self.data[0].normalize_inplace(module, scratch);
        }

        let mut coordinate_inv: Coordinate<Vec<u8>> = Coordinate::alloc(
            module,
            basek,
            address.k(),
            address.rows(),
            rank,
            &address.at(0).base1d.clone(),
        );

        for i in (0..address.n2() - 1).rev() {
            // Index polynomial X^{i}
            let coordinate: &Coordinate<Vec<u8>> = address.at(i + 1);

            coordinate_inv.base1d = coordinate.base1d.clone();

            // Inverts coordinate: X^{i} -> X^{-i}
            coordinate_inv.invert(
                module,
                coordinate,
                auto_keys.get(&-1).unwrap(),
                tensor_key,
                scratch,
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
                    let poly_lo: &mut GLWECiphertext<Vec<u8>> = &mut result_lo[j];

                    coordinate_inv.product_inplace(module, poly_lo, scratch);

                    chunk.iter_mut().for_each(|poly_hi| {
                        // Extract the first coefficient poly_lo
                        // tmp_a = TRACE([a, b, c, d]) -> [a, 0, 0, 0]
                        let (tmp_a_data, scratch_1) = scratch.tmp_vec_znx(module, rank + 1, size);
                        let mut tmp_a: GLWECiphertext<&mut [u8]> = GLWECiphertext {
                            data: tmp_a_data,
                            k: k_ct,
                            basek: basek,
                        };
                        tmp_a.trace::<Vec<u8>, _>(module, 0, log_n, poly_lo, auto_keys, scratch_1);

                        // Zeroes the first coefficient of poly_hi
                        // poly_hi = [a, b, c, d] - TRACE([a, b, c, d]) = [0, b, c, d]
                        let (tmp_b_data, scratch_2) = scratch_1.tmp_vec_znx(module, rank + 1, size);
                        let mut tmp_b: GLWECiphertext<&mut [u8]> = GLWECiphertext {
                            data: tmp_b_data,
                            k: k_ct,
                            basek: basek,
                        };
                        tmp_b.trace::<Vec<u8>, _>(module, 0, log_n, poly_hi, auto_keys, scratch_2);
                        poly_hi.sub_inplace_ab(module, &tmp_b);

                        // Adds extracted coefficient of poly_lo on poly_hi
                        // [a, 0, 0, 0] + [0, b, c, d]
                        poly_hi.add_inplace(module, &tmp_a);
                        poly_hi.normalize_inplace(module, scratch_2);

                        // Cyclic shift poly_lo by X^-1
                        poly_hi.rotate_inplace(module, -1);
                    })
                });
        }

        // Apply the last reverse shift to the top of the tree.
        self.data.iter_mut().for_each(|poly_lo| {
            let coordinate: &Coordinate<Vec<u8>> = address.at(0);

            // Inverts coordinate: X^{i} -> X^{-i}
            coordinate_inv.invert(
                module,
                coordinate,
                auto_keys.get(&-1).unwrap(),
                tensor_key,
                scratch,
            );

            coordinate.product_inplace(module, poly_lo, scratch);
        });

        self.state = false;
    }
}
