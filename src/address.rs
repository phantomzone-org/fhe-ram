use core::{
    automorphism::AutomorphismKey,
    ggsw_ciphertext::GGSWCiphertext,
    glwe_ciphertext::GLWECiphertext,
    keys::{SecretKey, SecretKeyFourier},
    tensor_key::TensorKey,
};

use backend::{
    FFT64, MatZnxDft, MatZnxDftToMut, MatZnxDftToRef, Module, ScalarZnxDft, ScalarZnxDftToRef,
    Scratch, ScratchOwned, VecZnx, VecZnxToMut, VecZnxToRef, ZnxViewMut,
};
use itertools::izip;
use sampling::source::{Source, new_seed};

use crate::parameters::Parameters;

pub struct Address {
    coordinates: Vec<Coordinate<Vec<u8>>>,
    k: usize,
    rows: usize,
    base2d: Base2D,
}

impl Address {
    pub fn alloc(params: &Parameters) -> Self {
        let max_addr: usize = params.max_addr();
        let decomp_n: Vec<u8> = params.decomp_n();

        let base_2d: Base2D = get_base_2d(max_addr as u32, decomp_n);
        let module: &Module<FFT64> = &params.module();
        let basek: usize = params.basek();
        let k: usize = params.k_addr();
        let rows: usize = params.rows_ct();
        let rank: usize = params.rank();

        let mut coordinates: Vec<Coordinate<Vec<u8>>> = Vec::new();
        base_2d.0.iter().for_each(|base1d| {
            coordinates.push(Coordinate::alloc(module, basek, k, rows, rank, base1d))
        });
        Self {
            coordinates: coordinates,
            k,
            rows,
            base2d: base_2d.clone(),
        }
    }

    pub fn encrypt_sk(&mut self, params: &Parameters, value: u32, sk: &SecretKey<Vec<u8>>) {
        debug_assert!(self.base2d.max() > value as usize);

        let module: &Module<FFT64> = &params.module();
        let rank: usize = params.rank();
        let size: usize = (params.k_evk() + params.basek() - 1) / params.basek();
        let sigma: f64 = params.xe();

        let mut scratch: ScratchOwned =
            ScratchOwned::new(GGSWCiphertext::encrypt_sk_scratch_space(module, rank, size));

        let mut sk_dft: SecretKeyFourier<Vec<u8>, FFT64> = SecretKeyFourier::alloc(module, rank);
        sk_dft.dft(module, sk);

        let mut source_xa: Source = Source::new(new_seed());
        let mut source_xe: Source = Source::new(new_seed());

        let mut remain: usize = value as _;
        izip!(self.coordinates.iter_mut(), self.base2d.0.iter()).for_each(|(coordinate, base1d)| {
            let max: usize = base1d.max();
            let k: usize = remain & (max - 1);
            coordinate.encrypt_sk(
                -(k as i64),
                module,
                &sk_dft,
                &mut source_xa,
                &mut source_xe,
                sigma,
                scratch.borrow(),
            );
            remain /= max;
        })
    }

    pub(crate) fn n2(&self) -> usize {
        self.coordinates.len()
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
}

pub(crate) struct Coordinate<D> {
    pub(crate) value: Vec<GGSWCiphertext<D, FFT64>>,
    pub(crate) base1d: Base1D,
}

impl Coordinate<Vec<u8>> {
    pub(crate) fn alloc(
        module: &Module<FFT64>,
        basek: usize,
        k: usize,
        rows: usize,
        rank: usize,
        base1d: &Base1D,
    ) -> Self {
        let mut coordinates: Vec<GGSWCiphertext<Vec<u8>, FFT64>> = Vec::new();
        base1d
            .0
            .iter()
            .for_each(|_| coordinates.push(GGSWCiphertext::alloc(module, basek, k, rows, rank)));
        Self {
            value: coordinates,
            base1d: base1d.clone(),
        }
    }

    pub(crate) fn invert_scratch_space(params: &Parameters) -> usize {
        GGSWCiphertext::automorphism_scratch_space(
            params.module(),
            params.size_addr(),
            params.size_addr(),
            params.size_evk(),
            params.size_evk(),
            params.rank(),
        )
    }

    pub(crate) fn product_scratch_space(params: &Parameters) -> usize {
        let module: &Module<FFT64> = params.module();
        let size_glwe: usize = (params.k_ct() + params.basek() - 1) / params.basek();
        let rank: usize = params.rank();
        let ggsw_size: usize = (params.k_evk() + params.basek() - 1) / params.basek();
        GLWECiphertext::external_product_scratch_space(
            module, size_glwe, rank, size_glwe, ggsw_size,
        ) | GLWECiphertext::external_product_scratch_space(
            module, size_glwe, rank, size_glwe, ggsw_size,
        )
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
        let gap: usize = 1; // self.base1d.gap(module.log_n());

        let mut remain: usize = value.abs() as usize;
        let mut tot_base: u8 = 0;

        izip!(self.value.iter_mut(), self.base1d.0.iter()).for_each(|(coordinate, base)| {
            let mask: usize = (1 << base) - 1;

            let chunk: usize = ((remain & mask) << tot_base) * gap;

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
        sk: &SecretKey<Vec<u8>>,
    ) where
        MatZnxDft<DataOther, FFT64>: MatZnxDftToRef<FFT64>,
        MatZnxDft<DataAK, FFT64>: MatZnxDftToRef<FFT64>,
        MatZnxDft<DataTK, FFT64>: MatZnxDftToRef<FFT64>,
    {
        assert!(auto_key.p() == -1);
        assert_eq!(
            self.value.len(),
            other.value.len(),
            "self.value.len(): {} != other.value.len(): {}",
            self.value.len(),
            other.value.len()
        );

        let mut sk_dft: SecretKeyFourier<Vec<u8>, FFT64> =
            SecretKeyFourier::alloc(module, sk.rank());
        sk_dft.dft(module, sk);

        izip!(self.value.iter_mut(), other.value.iter()).for_each(|(value, other)| {
            value.automorphism(module, other, auto_key, tensor_key, scratch);
        });
        self.base1d = other.base1d.clone();
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
        self.0.iter().for_each(|base| {
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
        let mut v: Vec<u8> = Vec::new();
        for i in 0..base.len() {
            if base[i] as u32 <= value_bit_size {
                v.push(base[i]);
                value_bit_size -= base[i] as u32;
            } else {
                if value_bit_size != 0 {
                    v.push(value_bit_size as u8);
                }
                out.push(Base1D(v));
                break 'outer;
            }
        }

        out.push(Base1D(v))
    }
    Base2D(out)
}
