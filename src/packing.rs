use core::glwe_ciphertext::GLWECiphertext;

use backend::{Module, Scratch, VecZnx, VecZnxToMut, VecZnxToRef, FFT64};

pub(crate) struct StreamPacker {
    basek: usize,
    accumulators: Vec<Accumulator>,
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
    pub(crate) fn new(module: &Module<FFT64>, basek: usize, k: usize, rank: usize) -> Self {
        let mut accumulators: Vec<Accumulator> = Vec::<Accumulator>::new();
        let log_n: usize = module.log_n();
        (0..log_n).for_each(|_| accumulators.push(Accumulator::alloc(module, basek, k, rank)));
        Self {
            basek: basek,
            accumulators: accumulators,
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

    pub(crate) fn add<DataRes, DataA>(
        &mut self,
        module: &Module<FFT64>,
        res: &mut Vec<GLWECiphertext<DataRes>>,
        a: Option<&GLWECiphertext<DataA>>,
        scratch: &mut Scratch,
    ) where
        VecZnx<DataRes>: VecZnxToMut + VecZnxToRef,
        VecZnx<DataA>: VecZnxToRef,
    {
        pack_core(
            module,
            self.basek,
            a,
            &mut self.accumulators,
            0,
            scratch,
        );
        self.counter += 1;
        if self.counter == module.n() {
            res.push(self.accumulators[module.log_n() - 1].data.clone());
            self.reset();
        }
    }

    pub(crate) fn flush<DataRes>(&mut self, module: &Module<FFT64>, res: &mut Vec<GLWECiphertext<DataRes>>)
    where
        VecZnx<DataRes>: VecZnxToMut + VecZnxToRef,
    {
        if self.counter != 0 {
            while self.counter != module.n() - 1 {
                self.add(module, res, None);
            }
        }
    }
}

fn pack_core<D>(
    module: &Module<FFT64>,
    a: Option<&GLWECiphertext<D>>,
    accumulators: &mut [Accumulator],
    i: usize,
    scratch: &mut Scratch,
) where VecZnx<D>: VecZnxToRef{
    let log_n = module.log_n();

    if i == log_n {
        return;
    }

    let (acc_prev, acc_next) = accumulators.split_at_mut(1);

    if !acc_prev[0].control {
        let acc_mut_ref: &mut Accumulator = &mut acc_prev[0]; // from split_at_mut

        if let Some(a_ref) = a {
            acc_mut_ref.data.copy_from(a_ref);
            acc_mut_ref.value = true
        } else {
            acc_mut_ref.value = false
        }
        acc_mut_ref.control = true;
    } else {
        combine(module, &mut acc_prev[0], a, i, scratch);

        acc_prev[0].control = false;

        if acc_prev[0].value {
            pack_core(
                module,
                Some(&acc_prev[0].data),
                acc_next,
                i + 1,
                scratch,
            );
        } else {
            pack_core(
                module,
                None,
                acc_next,
                i + 1,
                scratch,
            );
        }
    }
}

fn combine<D>(
    module: &Module<FFT64>,
    acc: &mut Accumulator,
    b: Option<&GLWECiphertext<D>>,
    i: usize,
    scratch: &mut Scratch,
) where VecZnx<D>: VecZnxToRef {


}