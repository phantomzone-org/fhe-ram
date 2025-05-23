use core::{
    automorphism::AutomorphismKey,
    ggsw_ciphertext::GGSWCiphertext,
    glwe_ciphertext::GLWECiphertext,
    keys::SecretKeyFourier,
    tensor_key::{self, TensorKey},
};

use backend::{
    FFT64, MatZnxDft, MatZnxDftToMut, MatZnxDftToRef, Module, ScalarZnxDft, ScalarZnxDftToRef,
    Scratch, VecZnx, VecZnxToMut, VecZnxToRef, ZnxViewMut,
};
use itertools::izip;
use sampling::source::Source;

pub struct Address<D> {
    coordinates: Vec<Coordinate<D>>,
    k: usize,
    rank: usize,
    rows: usize,
    decomp: Decomp,
}

impl Address<Vec<u8>> {
    pub fn new(
        module: &Module<FFT64>,
        decomp: &Decomp,
        basek: usize,
        k: usize,
        rows: usize,
        rank: usize,
    ) -> Self {
        let mut coordinates: Vec<Coordinate<Vec<u8>>> = Vec::new();
        (0..decomp.n1()).for_each(|_| {
            coordinates.push(Coordinate::alloc(module, basek, k, rows, rank, decomp))
        });
        Self {
            coordinates: coordinates,
            k,
            rank,
            rows,
            decomp: decomp.clone(),
        }
    }
}

impl<D> Address<D>
where
    MatZnxDft<D, FFT64>: MatZnxDftToMut<FFT64>,
{
    pub fn encrypt_sk<DataSk>(
        &mut self,
        idx: u32,
        module: &Module<FFT64>,
        sk_dft: &SecretKeyFourier<DataSk, FFT64>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        scratch: &mut Scratch,
    ) where
        ScalarZnxDft<DataSk, FFT64>: ScalarZnxDftToRef<FFT64>,
    {
        debug_assert!(self.decomp.max() > idx as usize);
        let max_n1: usize = self.decomp.max_n1();
        let mask_n1: usize = max_n1 - 1;
        let mut remain: usize = idx as _;
        self.coordinates.iter_mut().for_each(|coordinate| {
            let k: usize = remain & mask_n1;
            coordinate.encrypt_sk(
                -(k as i64),
                module,
                sk_dft,
                source_xa,
                source_xe,
                sigma,
                scratch,
            );
            remain /= max_n1;
        })
    }
}

impl<D> Address<D>
where
    MatZnxDft<D, FFT64>: MatZnxDftToRef<FFT64>,
{
    pub(crate) fn n2(&self) -> usize {
        self.coordinates.len()
    }

    pub(crate) fn n1(&self, idx: usize) -> usize {
        assert!(idx < self.coordinates.len());
        self.coordinates[idx].value.len()
    }

    pub(crate) fn at(&self, i: usize) -> &Coordinate<D> {
        &self.coordinates[i]
    }

    pub(crate) fn k(&self) -> usize {
        self.k
    }

    pub(crate) fn rows(&self) -> usize {
        self.rows
    }

    pub(crate) fn rank(&self) -> usize {
        self.rank
    }

    pub(crate) fn decomp(&self) -> Decomp {
        self.decomp.clone()
    }
}

pub(crate) struct Coordinate<D> {
    pub(crate) value: Vec<GGSWCiphertext<D, FFT64>>,
    pub(crate) decomp: Vec<u8>,
    pub(crate) gap: usize,
}

impl Coordinate<Vec<u8>> {
    pub fn alloc(
        module: &Module<FFT64>,
        basek: usize,
        k: usize,
        rows: usize,
        rank: usize,
        decomp: &Decomp,
    ) -> Self {
        let mut coordinates: Vec<GGSWCiphertext<Vec<u8>, FFT64>> = Vec::new();
        (0..decomp.n2())
            .for_each(|_| coordinates.push(GGSWCiphertext::alloc(module, basek, k, rows, rank)));
        Self {
            value: coordinates,
            decomp: decomp.base.clone(),
            gap: decomp.gap(module.log_n()),
        }
    }
}

impl<D> Coordinate<D> {
    pub fn n2(&self) -> usize {
        self.value.len()
    }
}

impl<D> Coordinate<D>
where
    MatZnxDft<D, FFT64>: MatZnxDftToMut<FFT64>,
{
    pub fn encrypt_sk<DataSk>(
        &mut self,
        value: i64,
        module: &Module<FFT64>,
        sk_dft: &SecretKeyFourier<DataSk, FFT64>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        scratch: &mut Scratch,
    ) where
        ScalarZnxDft<DataSk, FFT64>: ScalarZnxDftToRef<FFT64>,
    {
        let n: usize = module.n();
        let (mut scalar, scratch1) = scratch.tmp_scalar_znx(module, 1);
        let sign: i64 = value.signum();

        let mut remain: usize = value.abs() as usize;
        let mut tot_base: u8 = 0;
        izip!(self.value.iter_mut(), self.decomp.iter()).for_each(|(coordinate, base)| {
            let mask: usize = (1 << base) - 1;

            let chunk: usize = ((remain & mask) << tot_base) * self.gap;

            if sign < 0 && chunk != 0 {
                scalar.raw_mut()[n - chunk] = -1; // (X^i)^-1 = X^{2n-i} = -X^{n-i}
            } else {
                scalar.raw_mut()[chunk] = 1;
            }

            coordinate.encrypt_sk(
                module, &scalar, sk_dft, source_xa, source_xe, sigma, scratch1,
            );

            if sign < 0 && chunk != 0 {
                scalar.raw_mut()[n - chunk] = 0;
            } else {
                scalar.raw_mut()[chunk] = 0;
            }

            remain >>= base;
            tot_base += base;
        });
    }

    pub(crate) fn invert<DataOther, DataAK, DataTK>(
        &mut self,
        module: &Module<FFT64>,
        other: &Coordinate<DataOther>,
        auto_key: &AutomorphismKey<DataAK, FFT64>,
        tensor_key: &TensorKey<DataTK, FFT64>,
        scratch: &mut Scratch,
    ) where
        MatZnxDft<DataOther, FFT64>: MatZnxDftToRef<FFT64>,
        MatZnxDft<DataAK, FFT64>: MatZnxDftToRef<FFT64>,
        MatZnxDft<DataTK, FFT64>: MatZnxDftToRef<FFT64>,
    {
        assert!(auto_key.p() == -1);
        self.value.iter_mut().for_each(|value| {
            value.automorphism_inplace(module, auto_key, tensor_key, scratch);
        });
        self.decomp = other.decomp.clone();
        self.gap = other.gap;
    }
}

impl<D> Coordinate<D>
where
    MatZnxDft<D, FFT64>: MatZnxDftToRef<FFT64>,
{
    pub fn product<DataRes, DataA>(
        &self,
        module: &Module<FFT64>,
        res: &mut GLWECiphertext<DataRes>,
        a: &GLWECiphertext<DataA>,
        scratch: &mut Scratch,
    ) where
        VecZnx<DataRes>: VecZnxToMut,
        VecZnx<DataA>: VecZnxToRef,
    {
        self.value.iter().enumerate().for_each(|(i, coordinate)| {
            if i == 0 {
                res.external_product(module, a, coordinate, scratch);
            } else {
                res.external_product_inplace(module, coordinate, scratch);
            }
        });
    }

    pub fn product_inplace<DataRes>(
        &self,
        module: &Module<FFT64>,
        res: &mut GLWECiphertext<DataRes>,
        scratch: &mut Scratch,
    ) where
        VecZnx<DataRes>: VecZnxToMut,
    {
        self.value.iter().for_each(|coordinate| {
            res.external_product_inplace(module, coordinate, scratch);
        });
    }
}

#[derive(Clone, Debug)]
pub struct Decomp {
    pub n1: usize,
    pub n2: usize,
    pub base: Vec<u8>,
}

impl Decomp {
    pub fn n1(&self) -> usize {
        self.n1
    }

    pub fn n2(&self) -> usize {
        self.n2
    }

    pub fn max_n1(&self) -> usize {
        let mut max: usize = 1;
        self.base.iter().for_each(|i| max <<= i);
        max
    }

    pub fn max(&self) -> usize {
        let max_n1: usize = self.max_n1();
        let mut max: usize = 1;
        for _ in 0..self.n2() {
            max *= max_n1
        }
        max
    }

    pub fn gap(&self, log_n: usize) -> usize {
        let mut gap: usize = log_n;
        self.base.iter().for_each(|i| gap >>= i);
        1 << gap
    }

    pub fn basis_1d(&self) -> Vec<u8> {
        let n1: usize = self.n1();
        let n2: usize = self.n2();
        let mut decomp: Vec<u8> = vec![0u8; n1 * n2];
        for i in 0..n2 {
            decomp[i * n1..(i + 1) * n1].copy_from_slice(&self.base);
        }
        decomp
    }

    pub fn basis_2d(&self) -> Vec<Vec<u8>> {
        let mut decomp: Vec<Vec<u8>> = Vec::new();
        for _ in 0..self.n1() {
            decomp.push(self.base.clone());
        }
        decomp
    }
}
