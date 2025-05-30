use core::{AutomorphismKey, GGSWCiphertext, GLWECiphertext, GLWESecret, TensorKey};

use backend::{FFT64, Module, Scratch, ScratchOwned, ZnxViewMut};
use itertools::izip;
use sampling::source::{Source, new_seed};

use crate::parameters::Parameters;

/// [Address] stores GGSW(X^{addr}) in decomposed
/// form. That is, given addr = prod X^{a_i}, then
/// it stores Vec<[Coordinate]:(X^{a_0}), [Coordinate]:(X^{a_1}), ...>.
/// where [a_0, a_1, ...] is the representation in base N of a.
///
/// Such decomposition is necessary if the ring degree
/// N is smaller than the maximum supported address.
pub struct Address {
    coordinates: Vec<Coordinate<Vec<u8>>>,
    k: usize,
    rows: usize,
    base2d: Base2D,
}

impl Address {
    /// Allocates a new [Address].
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

    /// Encrypts an u32 value into an [Address] under the provided secret.
    pub fn encrypt_sk(&mut self, params: &Parameters, value: u32, sk: &GLWESecret<Vec<u8>, FFT64>) {
        debug_assert!(self.base2d.max() > value as usize);

        let module: &Module<FFT64> = params.module();
        let basek: usize = params.basek();
        let rank: usize = params.rank();
        let k: usize = params.k_addr();
        let sigma: f64 = params.xe();

        let mut scratch: ScratchOwned = ScratchOwned::new(
            GGSWCiphertext::encrypt_sk_scratch_space(module, basek, k, rank),
        );

        let mut source_xa: Source = Source::new(new_seed());
        let mut source_xe: Source = Source::new(new_seed());

        let mut remain: usize = value as _;
        izip!(self.coordinates.iter_mut(), self.base2d.0.iter()).for_each(|(coordinate, base1d)| {
            let max: usize = base1d.max();
            let k: usize = remain & (max - 1);
            coordinate.encrypt_sk(
                -(k as i64),
                module,
                &sk,
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

/// Coordinate stores Vec<GGSW(X^a_i)> such that prod X^{a_i} = a.
/// This provides a second decomposition over the one in base N to
/// to ensure that the digits are small enough to enable HE operation
/// over the digits (e.g. 2-4 bits digits instead of log(N)-bits digits).
pub(crate) struct Coordinate<D> {
    pub(crate) value: Vec<GGSWCiphertext<D, FFT64>>,
    pub(crate) base1d: Base1D,
}

impl Coordinate<Vec<u8>> {
    /// Allocates a new [Coordinate].
    ///
    /// # Arguments
    ///
    /// * `module`: pre-computed FFT tables.
    /// * `basek`: base 2 logarithm of the
    /// * `rows`: number of digits for the key-switching decomposition.
    /// * `rank`: rank of the GLWE/GGLE/GGSW ciphertexts.
    /// * `base1d`: digit decomposition of the coordinate (e.g. [12], [6, 6], [4, 4, 4] or [3, 3, 3, 3] for LogN = 12).
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

    /// Scratch space required to invert a coordinate, i.e. map GGSW(X^{i}) to GGSW(X^{-i}).
    pub(crate) fn invert_scratch_space(params: &Parameters) -> usize {
        GGSWCiphertext::automorphism_scratch_space(
            params.module(), // FFT/NTT Tables.
            params.basek(),  // Torus base 2 decomposition.
            params.k_addr(), // Output GGSW Torus precision.
            params.k_addr(), // Input GGSW Torus precision.
            params.k_evk(),  // Automorphism GLWE Torus precision.
            params.k_evk(),  // Tensor GLWE Torus precision.
            params.rank(),   // GLWE/GGLWE/GGSW rank.
        )
    }

    /// Scratch space required to evaluate GGSW(X^{i}) * GLWE(m).
    pub(crate) fn product_scratch_space(params: &Parameters) -> usize {
        GLWECiphertext::external_product_scratch_space(
            params.module(), // FFT/NTT Tables.
            params.basek(),  // Torus base 2 decomposition.
            params.k_ct(),   // Output GLWE Torus precision.
            params.k_ct(),   // Input GLWE Torus precision.
            params.k_addr(), // Address GGSW Torus precision.
            params.rank(),   // GLWE/GGSW rank.
        ) | GLWECiphertext::external_product_inplace_scratch_space(
            params.module(), // FFT/NTT Tables.
            params.basek(),  // Torus base 2 decomposition.
            params.k_ct(),   // Input/Output GLWE Torus precision.
            params.k_addr(), // Address GGSW Torus precision.
            params.rank(),   // GLWE/GGSW rank.
        )
    }
}

impl<D: AsMut<[u8]> + AsRef<[u8]>> Coordinate<D> {
    /// Encrypts a value in [-N+1, N-1] as GGSW(X^{value}).
    ///
    /// #Arguments
    ///
    /// * `value`: value to encrypt.
    /// * `module`: FFT/NTT tables.
    /// * `sk_dft`: secret in Fourier domain.
    /// * `source_xa`: random coins generator for public polynomials.
    /// * `source_xe`: random coins generator for noise.
    /// * `sigma`: standard deviation of the noise.
    /// * `scratch`: scratch space provider.
    pub(crate) fn encrypt_sk<DataSk: AsRef<[u8]>>(
        &mut self,
        value: i64,
        module: &Module<FFT64>,
        sk: &GLWESecret<DataSk, FFT64>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        scratch: &mut Scratch,
    ) {
        let n: usize = module.n();

        assert!(value.abs() < n as i64);

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

            coordinate.encrypt_sk(module, &scalar, sk, source_xa, source_xe, sigma, scratch1);

            if sign < 0 && chunk != 0 {
                scalar.raw_mut()[n - chunk] = 0;
            } else {
                scalar.raw_mut()[chunk] = 0;
            }

            remain >>= base;
            tot_base += base;
        });
    }

    /// Maps GGSW(X^{i}) to GGSW(X^{-i}).
    ///
    /// #Arguments
    ///
    /// * `module`: FFT/NTT tables.
    /// * `other`: coordinate to invert.
    /// * `auto_key`: GGLWE(AUTO(s, -1)).
    /// * `tensor_key`: GGLWE(TENSOR(s)).
    /// * `scratch`: scratch space provider.
    pub(crate) fn invert<DataOther: AsRef<[u8]>, DataAK: AsRef<[u8]>, DataTK: AsRef<[u8]>>(
        &mut self,
        module: &Module<FFT64>,
        other: &Coordinate<DataOther>,
        auto_key: &AutomorphismKey<DataAK, FFT64>,
        tensor_key: &TensorKey<DataTK, FFT64>,
        scratch: &mut Scratch,
    ) {
        assert!(auto_key.p() == -1);
        assert_eq!(
            self.value.len(),
            other.value.len(),
            "self.value.len(): {} != other.value.len(): {}",
            self.value.len(),
            other.value.len()
        );

        izip!(self.value.iter_mut(), other.value.iter()).for_each(|(value, other)| {
            value.automorphism(module, other, auto_key, tensor_key, scratch);
        });
        self.base1d = other.base1d.clone();
    }
}

impl<D: AsRef<[u8]>> Coordinate<D> {
    /// Evaluates GLWE(m) * GGSW(X^i).
    pub(crate) fn product<DataRes: AsMut<[u8]> + AsRef<[u8]>, DataA: AsRef<[u8]>>(
        &self,
        module: &Module<FFT64>,
        res: &mut GLWECiphertext<DataRes>,
        a: &GLWECiphertext<DataA>,
        scratch: &mut Scratch,
    ) {
        self.value.iter().enumerate().for_each(|(i, coordinate)| {
            if i == 0 {
                res.external_product(module, a, coordinate, scratch);
            } else {
                res.external_product_inplace(module, coordinate, scratch);
            }
        });
    }

    /// Evaluates GLWE(m) * GGSW(X^i).
    pub(crate) fn product_inplace<DataRes: AsMut<[u8]> + AsRef<[u8]>>(
        &self,
        module: &Module<FFT64>,
        res: &mut GLWECiphertext<DataRes>,
        scratch: &mut Scratch,
    ) {
        self.value.iter().for_each(|coordinate| {
            res.external_product_inplace(module, coordinate, scratch);
        });
    }
}

/// Helper for 1D digit decomposition.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Base1D(pub Vec<u8>);

impl Base1D {
    pub(crate) fn max(&self) -> usize {
        let mut max: usize = 1;
        self.0.iter().for_each(|i| max <<= i);
        max
    }

    #[allow(dead_code)]
    pub(crate) fn gap(&self, log_n: usize) -> usize {
        let mut gap: usize = log_n;
        self.0.iter().for_each(|i| gap >>= i);
        1 << gap
    }

    #[allow(dead_code)]
    pub(crate) fn decomp(&self, value: u32) -> Vec<u8> {
        let mut decomp: Vec<u8> = Vec::new();
        let mut sum_bases: u8 = 0;
        self.0.iter().for_each(|base| {
            decomp.push(((value >> sum_bases) & (1 << base) - 1) as u8);
            sum_bases += base;
        });
        decomp
    }

    #[allow(dead_code)]
    pub(crate) fn recomp(&self, decomp: &Vec<u8>) -> u32 {
        let mut value: u32 = 0;
        let mut sum_bases: u8 = 0;
        self.0.iter().enumerate().for_each(|(i, base)| {
            value |= (decomp[i] << sum_bases) as u32;
            sum_bases += base;
        });
        value
    }
}

/// Helpe for 2D digit decomposition.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Base2D(pub Vec<Base1D>);

impl Base2D {
    pub(crate) fn max(&self) -> usize {
        self.as_1d().max()
    }

    pub(crate) fn as_1d(&self) -> Base1D {
        Base1D(
            self.0
                .iter()
                .flat_map(|array| array.0.iter().map(|&x| x))
                .collect(),
        )
    }

    #[allow(dead_code)]
    pub(crate) fn decomp(&self, value: u32) -> Vec<u8> {
        self.as_1d().decomp(value)
    }

    #[allow(dead_code)]
    pub(crate) fn recomp(&self, decomp: &Vec<u8>) -> u32 {
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
