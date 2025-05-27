use core::{automorphism::AutomorphismKey, elem::Infos, glwe_ciphertext::GLWECiphertext};
use std::collections::HashMap;

use backend::{
    FFT64, MatZnxDft, MatZnxDftToRef, Module, Scratch, VecZnx, VecZnxAlloc, VecZnxToRef,
};

pub(crate) struct StreamPacker {
    accumulators: Vec<Accumulator>,
    log_batch: usize,
    counter: usize,
}

pub(crate) struct Accumulator {
    data: GLWECiphertext<Vec<u8>>,
    value: bool,
    control: bool,
}

impl Accumulator {
    pub(crate) fn alloc(module: &Module<FFT64>, basek: usize, k: usize, rank: usize) -> Self {
        Self {
            data: GLWECiphertext::alloc(module, basek, k, rank),
            value: false,
            control: false,
        }
    }
}

impl StreamPacker {
    pub(crate) fn new(
        module: &Module<FFT64>,
        log_batch: usize,
        basek: usize,
        k: usize,
        rank: usize,
    ) -> Self {
        let mut accumulators: Vec<Accumulator> = Vec::<Accumulator>::new();
        let log_n: usize = module.log_n();
        (0..log_n - log_batch)
            .for_each(|_| accumulators.push(Accumulator::alloc(module, basek, k, rank)));
        Self {
            accumulators: accumulators,
            log_batch,
            counter: 0,
        }
    }

    pub(crate) fn reset(&mut self) {
        for i in 0..self.accumulators.len() {
            self.accumulators[i].value = false;
            self.accumulators[i].control = false;
        }
        self.counter = 0;
    }

    pub(crate) fn add_scratch_space(
        module: &Module<FFT64>,
        ct_size: usize,
        autokey_size: usize,
        rank: usize,
    ) -> usize {
        pack_core_scratch_space(module, ct_size, autokey_size, rank)
    }

    pub(crate) fn add<DataA, DataAK>(
        &mut self,
        module: &Module<FFT64>,
        res: &mut Vec<GLWECiphertext<Vec<u8>>>,
        a: Option<&GLWECiphertext<DataA>>,
        auto_keys: &HashMap<i64, AutomorphismKey<DataAK, FFT64>>,
        scratch: &mut Scratch,
    ) where
        VecZnx<DataA>: VecZnxToRef,
        MatZnxDft<DataAK, FFT64>: MatZnxDftToRef<FFT64>,
    {
        pack_core(
            module,
            a,
            &mut self.accumulators,
            self.log_batch,
            auto_keys,
            scratch,
        );
        self.counter += 1 << self.log_batch;
        if self.counter == module.n() {
            res.push(
                self.accumulators[module.log_n() - self.log_batch - 1]
                    .data
                    .clone(),
            );
            self.reset();
        }
    }

    pub(crate) fn flush<DataAK>(
        &mut self,
        module: &Module<FFT64>,
        res: &mut Vec<GLWECiphertext<Vec<u8>>>,
        auto_keys: &HashMap<i64, AutomorphismKey<DataAK, FFT64>>,
        scratch: &mut Scratch,
    ) where
        MatZnxDft<DataAK, FFT64>: MatZnxDftToRef<FFT64>,
    {
        if self.counter != 0 {
            while self.counter != 0 {
                self.add(
                    module,
                    res,
                    None::<&GLWECiphertext<Vec<u8>>>,
                    auto_keys,
                    scratch,
                );
            }
        }
    }
}

fn pack_core_scratch_space(
    module: &Module<FFT64>,
    ct_size: usize,
    autokey_size: usize,
    rank: usize,
) -> usize {
    combine_scratch_space(module, ct_size, autokey_size, rank)
}

fn pack_core<D, DataAK>(
    module: &Module<FFT64>,
    a: Option<&GLWECiphertext<D>>,
    accumulators: &mut [Accumulator],
    i: usize,
    auto_keys: &HashMap<i64, AutomorphismKey<DataAK, FFT64>>,
    scratch: &mut Scratch,
) where
    VecZnx<D>: VecZnxToRef,
    MatZnxDft<DataAK, FFT64>: MatZnxDftToRef<FFT64>,
{
    let log_n: usize = module.log_n();

    if i == log_n {
        return;
    }

    let (acc_prev, acc_next) = accumulators.split_at_mut(1);

    if !acc_prev[0].control {
        let acc_mut_ref: &mut Accumulator = &mut acc_prev[0]; // from split_at_mut

        if let Some(a_ref) = a {
            acc_mut_ref.data.copy(module, a_ref);
            acc_mut_ref.value = true
        } else {
            acc_mut_ref.value = false
        }
        acc_mut_ref.control = true;
    } else {
        combine(module, &mut acc_prev[0], a, i, auto_keys, scratch);

        acc_prev[0].control = false;

        if acc_prev[0].value {
            pack_core::<Vec<u8>, _>(
                module,
                Some(&acc_prev[0].data),
                acc_next,
                i + 1,
                auto_keys,
                scratch,
            );
        } else {
            pack_core(module, None, acc_next, i + 1, auto_keys, scratch);
        }
    }
}

fn combine_scratch_space(
    module: &Module<FFT64>,
    ct_size: usize,
    autokey_size: usize,
    rank: usize,
) -> usize {
    2 * module.bytes_of_vec_znx(rank + 1, ct_size)
        + (GLWECiphertext::rsh_scratch_space(module)
            | GLWECiphertext::automorphism_scratch_space(
                module,
                ct_size,
                rank,
                ct_size,
                autokey_size,
            ))
}

fn combine<D, DataAK>(
    module: &Module<FFT64>,
    acc: &mut Accumulator,
    b: Option<&GLWECiphertext<D>>,
    i: usize,
    auto_keys: &HashMap<i64, AutomorphismKey<DataAK, FFT64>>,
    scratch: &mut Scratch,
) where
    VecZnx<D>: VecZnxToRef,
    MatZnxDft<DataAK, FFT64>: MatZnxDftToRef<FFT64>,
{
    let log_n: usize = module.log_n();
    let a: &mut GLWECiphertext<Vec<u8>> = &mut acc.data;
    let basek: usize = a.basek();
    let k: usize = a.k();
    let rank: usize = a.rank();
    let cols: usize = rank + 1;
    let size: usize = a.size();

    let gal_el: i64;

    if i == 0 {
        gal_el = -1;
    } else {
        gal_el = module.galois_element(1 << (i - 1))
    }

    if acc.value {
        a.rsh(1, scratch);

        if let Some(b) = b {
            let (tmp_b_data, scratch_1) = scratch.tmp_vec_znx(module, cols, size);
            let mut tmp_b: GLWECiphertext<&mut [u8]> = GLWECiphertext {
                data: tmp_b_data,
                k: k,
                basek: basek,
            };

            {
                let (tmp_a_data, scratch_2) = scratch_1.tmp_vec_znx(module, cols, size);
                let mut tmp_a: GLWECiphertext<&mut [u8]> = GLWECiphertext {
                    data: tmp_a_data,
                    k: k,
                    basek: basek,
                };

                // tmp_a = b * X^t
                tmp_a.rotate(module, 1 << (log_n - i - 1), b);

                // tmp_a >>= 1
                tmp_a.rsh(1, scratch_2);

                // tmp_b = a - b*X^t
                tmp_b.sub(module, a, &tmp_a);
                tmp_b.normalize_inplace(module, scratch_2);

                // a = a + b * X^t
                a.add_inplace(module, &tmp_a);
            }

            // tmp_b = phi(a - b * X^t)
            if let Some(key) = auto_keys.get(&gal_el) {
                tmp_b.automorphism_inplace(module, key, scratch_1);
            } else {
                panic!("auto_key[{}] not found", gal_el);
            }

            // a = X^t(a*X^-t + b + phi(aX^-t - b))
            a.add_inplace(module, &tmp_b);
            a.normalize_inplace(module, scratch_1);
        } else {
            // a = a + phi(a)
            if let Some(key) = auto_keys.get(&gal_el) {
                a.automorphism_add_inplace(module, key, scratch);
            } else {
                panic!("auto_key[{}] not found", gal_el);
            }
        }
    } else {
        if let Some(b) = b {
            let (tmp_b_data, scratch_1) = scratch.tmp_vec_znx(module, cols, size);
            let mut tmp_b: GLWECiphertext<&mut [u8]> = GLWECiphertext {
                data: tmp_b_data,
                k: k,
                basek: basek,
            };

            tmp_b.rotate(module, 1 << (log_n - i - 1), b);
            tmp_b.rsh(1, scratch_1);

            // a = (b* X^t - phi(b* X^t))
            if let Some(key) = auto_keys.get(&gal_el) {
                a.automorphism_sub_ba::<&mut [u8], _>(module, &tmp_b, key, scratch_1);
            } else {
                panic!("auto_key[{}] not found", gal_el);
            }

            acc.value = true;
        }
    }
}
