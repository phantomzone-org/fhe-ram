use core::{
    automorphism::AutomorphismKey, glwe_ciphertext::GLWECiphertext, glwe_plaintext::GLWEPlaintext,
    keys::SecretKeyFourier,
};
use std::collections::HashMap;

use backend::{
    Encoding, FFT64, MatZnxDft, MatZnxDftToRef, Module, ScalarZnxDft, ScalarZnxDftToRef, Scratch,
};
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
    pub fn new(
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

    pub fn encrypt_sk<DataSk>(
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

    pub fn read<DataAdr, DataAK>(
        &self,
        module: &Module<FFT64>,
        address: &Address<DataAdr>,
        auto_keys: &HashMap<i64, AutomorphismKey<DataAK, FFT64>>,
        scratch: &mut Scratch,
    ) -> GLWECiphertext<Vec<u8>>
    where
        MatZnxDft<DataAdr, FFT64>: MatZnxDftToRef<FFT64>,
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

        let log_n = module.log_n();

        let mut packer: StreamPacker = StreamPacker::new(module, basek, k_ct, rank);
        let mut results: Vec<GLWECiphertext<Vec<u8>>> = Vec::new();

        let mut tmp_ct: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(module, basek, k_ct, rank);

        for i in 0..address.n2() {
            let coordinate: &Coordinate<DataAdr> = address.at(i);

            let res_prev: &Vec<GLWECiphertext<Vec<u8>>>;

            if i == 0 {
                res_prev = &self.data
            } else {
                res_prev = &results
            }

            if i < address.n2() - 1 {
                let mut result_next = Vec::new();

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
}
