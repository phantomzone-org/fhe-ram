use core::{
    automorphism::AutomorphismKey, ggsw_ciphertext::GGSWCiphertext,
    glwe_ciphertext::GLWECiphertext, keys::SecretKeyFourier, tensor_key::TensorKey,
};

use backend::{
    FFT64, MatZnxDft, MatZnxDftToMut, MatZnxDftToRef, Module, ScalarZnxDft, ScalarZnxDftToRef,
    Scratch, VecZnx, VecZnxToMut, VecZnxToRef, ZnxViewMut,
};
use itertools::izip;
use sampling::source::Source;

pub struct Address {
    coordinates: Vec<Coordinate<Vec<u8>>>,
    k: usize,
    rank: usize,
    rows: usize,
    base2d: Base2D,
}

impl Address {
    pub(crate) fn new(
        module: &Module<FFT64>,
        base2d: &Base2D,
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
            base2d: base2d.clone(),
        }
    }

    pub(crate) fn encrypt_sk<DataSk>(
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

    pub(crate) fn n2(&self) -> usize {
        self.coordinates.len()
    }

    pub(crate) fn n1(&self, idx: usize) -> usize {
        assert!(idx < self.coordinates.len());
        self.coordinates[idx].value.len()
    }

    pub(crate) fn at(&self, i: usize) -> &Coordinate<Vec<u8>> {
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
    pub(crate) fn alloc(
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
    pub(crate) fn n2(&self) -> usize {
        self.value.len()
    }
}

impl<D> Coordinate<D>
where
    MatZnxDft<D, FFT64>: MatZnxDftToMut<FFT64>,
{
    pub(crate) fn encrypt_sk<DataSk>(
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
    pub(crate) fn product<DataRes, DataA>(
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

    pub(crate) fn product_inplace<DataRes>(
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
pub(crate) struct Decomp {
    pub n1: usize,
    pub n2: usize,
    pub base: Vec<u8>,
}

impl Decomp {
    pub(crate) fn n1(&self) -> usize {
        self.n1
    }

    pub(crate) fn n2(&self) -> usize {
        self.n2
    }

    pub(crate) fn max_n1(&self) -> usize {
        let mut max: usize = 1;
        self.base.iter().for_each(|i| max <<= i);
        max
    }

    pub(crate) fn max(&self) -> usize {
        let max_n1: usize = self.max_n1();
        let mut max: usize = 1;
        for _ in 0..self.n2() {
            max *= max_n1
        }
        max
    }

    pub(crate) fn gap(&self, log_n: usize) -> usize {
        let mut gap: usize = log_n;
        self.base.iter().for_each(|i| gap >>= i);
        1 << gap
    }

    pub(crate) fn basis_1d(&self) -> Vec<u8> {
        let n1: usize = self.n1();
        let n2: usize = self.n2();
        let mut decomp: Vec<u8> = vec![0u8; n1 * n2];
        for i in 0..n2 {
            decomp[i * n1..(i + 1) * n1].copy_from_slice(&self.base);
        }
        decomp
    }

    pub(crate) fn basis_2d(&self) -> Vec<Vec<u8>> {
        let mut decomp: Vec<Vec<u8>> = Vec::new();
        for _ in 0..self.n1() {
            decomp.push(self.base.clone());
        }
        decomp
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Base1D(pub Vec<u8>);

impl Base1D {
    pub fn max(&self) -> usize {
        let mut max: usize = 1;
        self.0.iter().for_each(|i| max <<= i);
        max
    }

    pub fn gap(&self, log_n: usize) -> usize {
        let mut gap: usize = log_n;
        self.0.iter().for_each(|i| gap >>= i);
        1 << gap
    }

    pub fn decomp(&self, value: u32) -> Vec<u8> {
        let mut decomp: Vec<u8> = Vec::new();
        let mut sum_bases: u8 = 0;
        self.0.iter().enumerate().for_each(|(i, base)| {
            decomp.push(((value >> sum_bases) & (1 << base) - 1) as u8);
            sum_bases += base;
        });
        decomp
    }

    pub fn recomp(&self, decomp: &Vec<u8>) -> u32 {
        let mut value: u32 = 0;
        let mut sum_bases: u8 = 0;
        self.0.iter().enumerate().for_each(|(i, base)| {
            value |= (decomp[i] << sum_bases) as u32;
            sum_bases += base;
        });
        value
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Base2D(pub Vec<Base1D>);

impl Base2D {
    pub fn max(&self) -> usize {
        self.as_1d().max()
    }

    pub fn as_1d(&self) -> Base1D {
        Base1D(
            self.0
                .iter()
                .flat_map(|array| array.0.iter().map(|&x| x))
                .collect(),
        )
    }

    pub fn decomp(&self, value: u32) -> Vec<u8> {
        self.as_1d().decomp(value)
    }

    pub fn recomp(&self, decomp: &Vec<u8>) -> u32 {
        self.as_1d().recomp(decomp)
    }
}

pub(crate) fn get_base_2d(value: u32, base: Vec<u8>) -> Base2D {
    let mut out: Vec<Base1D> = Vec::new();
    let mut value_bit_size: u32 = 32 - (value - 1).leading_zeros();

    'outer: while value_bit_size != 0 {
        let mut v = Vec::new();
        for i in 0..base.len() {
            if base[i] as u32 <= value_bit_size {
                v.push(base[i]);
                value_bit_size -= base[i] as u32;
            } else {
                v.push(value_bit_size as u8);
                out.push(Base1D(v));
                break 'outer;
            }
        }
        out.push(Base1D(v))
    }

    Base2D(out)
}