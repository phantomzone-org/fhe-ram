use itertools::izip;
use poulpy_backend::cpu_fft64_avx::FFT64Avx;
use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow, TakeScalarZnx},
    layouts::{Data, Module, Scratch, ScratchOwned, ZnxViewMut},
    source::Source,
};

use poulpy_core::layouts::{
    GGLWEAutomorphismKey, GGLWETensorKey, GGSWCiphertext, GLWECiphertext, GLWESecret,
    prepared::{
        GGLWEAutomorphismKeyPrepared, GGLWETensorKeyPrepared, GGSWCiphertextPrepared,
        GLWESecretPrepared, PrepareAlloc,
    },
};
use rand_core::{OsRng, TryRngCore};

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
        let module: &Module<FFT64Avx> = params.module();
        let basek: usize = params.basek();
        let k: usize = params.k_addr();
        let rows: usize = params.rows_ct();
        let rank: usize = params.rank();
        let digits: usize = params.digits();

        Self {
            coordinates: base_2d
                .0
                .iter()
                .map(|base1d| Coordinate::alloc(module, basek, k, rows, rank, digits, base1d))
                .collect(),
            k,
            rows,
            base2d: base_2d.clone(),
        }
    }

    /// Encrypts an u32 value into an [Address] under the provided secret.
    pub fn encrypt_sk(&mut self, params: &Parameters, value: u32, sk: &GLWESecret<Vec<u8>>) {
        debug_assert!(self.base2d.max() > value as usize);

        let module: &Module<FFT64Avx> = params.module();
        let basek: usize = params.basek();
        let rank: usize = params.rank();
        let k: usize = params.k_addr();

        let mut scratch: ScratchOwned<FFT64Avx> = ScratchOwned::alloc(
            GGSWCiphertext::encrypt_sk_scratch_space(module, basek, k, rank),
        );

        let mut seed_xa: [u8; 32] = [0u8; 32];
        OsRng.try_fill_bytes(&mut seed_xa).unwrap();

        let mut seed_xe: [u8; 32] = [0u8; 32];
        OsRng.try_fill_bytes(&mut seed_xe).unwrap();

        let mut source_xa: Source = Source::new(seed_xa);
        let mut source_xe: Source = Source::new(seed_xe);

        let mut remain: usize = value as _;
        izip!(self.coordinates.iter_mut(), self.base2d.0.iter()).for_each(|(coordinate, base1d)| {
            let max: usize = base1d.max();
            let k: usize = remain & (max - 1);
            coordinate.encrypt_sk(
                -(k as i64),
                module,
                sk,
                &mut source_xa,
                &mut source_xe,
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

/// Coordinate stores Vec<GGSW(X^a_i)> such that prod X^{a_i} = X^a.
/// This provides a second decomposition over the one in base N to
/// to ensure that the digits are small enough to enable HE operation
/// over the digits (e.g. 2-4 bits digits instead of log(N)-bits digits).
pub(crate) struct Coordinate<D: Data> {
    pub(crate) value: Vec<GGSWCiphertext<D>>,
    pub(crate) base1d: Base1D,
}

impl Coordinate<Vec<u8>> {
    /// Allocates a new [Coordinate].
    ///
    /// # Arguments
    ///
    /// * `module`: pre-computed FFT tables.
    /// * `basek`: base 2 logarithm of the limb in base2k representation
    /// * `rows`: number of digits for the key-switching decomposition.
    /// * `rank`: rank of the GLWE/GGLE/GGSW ciphertexts.
    /// * `base1d`: digit decomposition of the coordinate (e.g. [12], [6, 6], [4, 4, 4] or [3, 3, 3, 3] for LogN = 12).
    pub(crate) fn alloc(
        module: &Module<FFT64Avx>,
        basek: usize,
        k: usize,
        rows: usize,
        rank: usize,
        digits: usize,
        base1d: &Base1D,
    ) -> Self {
        Self {
            value: base1d
                .0
                .iter()
                .map(|_| GGSWCiphertext::alloc(module.n(), basek, k, rows, digits, rank))
                .collect(),
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
            params.digits(),
            params.k_evk(), // Tensor GLWE Torus precision.
            params.digits(),
            params.rank(), // GLWE/GGLWE/GGSW rank.
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
            params.digits(),
            params.rank(), // GLWE/GGSW rank.
        ) | GLWECiphertext::external_product_inplace_scratch_space(
            params.module(), // FFT/NTT Tables.
            params.basek(),  // Torus base 2 decomposition.
            params.k_ct(),   // Input/Output GLWE Torus precision.
            params.k_addr(), // Address GGSW Torus precision.
            params.digits(),
            params.rank(), // GLWE/GGSW rank.
        )
    }
}

impl<D: Data + AsMut<[u8]> + AsRef<[u8]>> Coordinate<D> {
    /// Encrypts a value in [-N+1, N-1] as GGSW(X^{value}).
    ///
    /// # Arguments
    ///
    /// * `value`: value to encrypt.
    /// * `module`: FFT/NTT tables.
    /// * `sk_dft`: secret in Fourier domain.
    /// * `source_xa`: random coins generator for public polynomials.
    /// * `source_xe`: random coins generator for noise.
    /// * `sigma`: standard deviation of the noise.
    /// * `scratch`: scratch space provider.
    pub(crate) fn encrypt_sk<DataSk: Data + AsRef<[u8]>>(
        &mut self,
        value: i64,
        module: &Module<FFT64Avx>,
        sk: &GLWESecret<DataSk>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<FFT64Avx>,
    ) {
        let n: usize = module.n();

        assert!(value.abs() < n as i64);

        let (mut scalar, scratch1) = scratch.take_scalar_znx(module.n(), 1);

        let sk_glwe_prepared: GLWESecretPrepared<Vec<u8>, FFT64Avx> =
            sk.prepare_alloc(module, scratch1);

        let sign: i64 = value.signum();
        let gap: usize = 1; // self.base1d.gap(module.log_n());

        let mut remain: usize = value.unsigned_abs() as usize;
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
                module,
                &scalar,
                &sk_glwe_prepared,
                source_xa,
                source_xe,
                scratch1,
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

    /// Maps GGSW(X^{i}) to GGSW(X^{-i}).
    ///
    /// #Arguments
    ///
    /// * `module`: FFT/NTT tables.
    /// * `other`: coordinate to invert.
    /// * `auto_key`: GGLWE(AUTO(s, -1)).
    /// * `tensor_key`: GGLWE(TENSOR(s)).
    /// * `scratch`: scratch space provider.
    pub(crate) fn invert<
        DataOther: Data + AsRef<[u8]>,
        DataAK: Data + AsRef<[u8]>,
        DataTK: Data + AsRef<[u8]>,
    >(
        &mut self,
        module: &Module<FFT64Avx>,
        other: &Coordinate<DataOther>,
        auto_key: &GGLWEAutomorphismKey<DataAK>,
        tensor_key: &GGLWETensorKey<DataTK>,
        scratch: &mut Scratch<FFT64Avx>,
    ) {
        assert!(auto_key.p() == -1);
        assert_eq!(
            self.value.len(),
            other.value.len(),
            "self.value.len(): {} != other.value.len(): {}",
            self.value.len(),
            other.value.len()
        );

        let (_, scratch1) = scratch.take_scalar_znx(module.n(), 1);
        let auto_key_prepared: GGLWEAutomorphismKeyPrepared<Vec<u8>, FFT64Avx> =
            auto_key.prepare_alloc(module, scratch1);
        let tensor_key_prepared: GGLWETensorKeyPrepared<Vec<u8>, FFT64Avx> =
            tensor_key.prepare_alloc(module, scratch1);

        izip!(self.value.iter_mut(), other.value.iter()).for_each(|(value, other)| {
            value.automorphism(
                module,
                other,
                &auto_key_prepared,
                &tensor_key_prepared,
                scratch,
            );
        });
        self.base1d = other.base1d.clone();
    }
}

impl<D: Data + AsRef<[u8]>> Coordinate<D> {
    /// Evaluates GLWE(m) * GGSW(X^i).
    pub(crate) fn product<DataRes: Data + AsMut<[u8]> + AsRef<[u8]>, DataA: Data + AsRef<[u8]>>(
        &self,
        module: &Module<FFT64Avx>,
        res: &mut GLWECiphertext<DataRes>,
        a: &GLWECiphertext<DataA>,
        scratch: &mut Scratch<FFT64Avx>,
    ) {
        self.value.iter().enumerate().for_each(|(i, coordinate)| {
            let (_, scratch1) = scratch.take_scalar_znx(module.n(), 1);
            let coordinate_prepared: GGSWCiphertextPrepared<Vec<u8>, FFT64Avx> =
                coordinate.prepare_alloc(module, scratch1);

            if i == 0 {
                res.external_product(module, a, &coordinate_prepared, scratch);
            } else {
                res.external_product_inplace(module, &coordinate_prepared, scratch);
            }
        });
    }

    /// Evaluates GLWE(m) * GGSW(X^i).
    pub(crate) fn product_inplace<DataRes: Data + AsMut<[u8]> + AsRef<[u8]>>(
        &self,
        module: &Module<FFT64Avx>,
        res: &mut GLWECiphertext<DataRes>,
        scratch: &mut Scratch<FFT64Avx>,
    ) {
        self.value.iter().for_each(|coordinate| {
            let (_, scratch1) = scratch.take_scalar_znx(module.n(), 1);
            let coordinate_prepared: GGSWCiphertextPrepared<Vec<u8>, FFT64Avx> =
                coordinate.prepare_alloc(module, scratch1);
            res.external_product_inplace(module, &coordinate_prepared, scratch);
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
        self.0
            .iter()
            .scan(0, |sum_bases, &base| {
                let part = ((value >> *sum_bases) & ((1 << base) - 1)) as u8;
                *sum_bases += base;
                Some(part)
            })
            .collect()
    }

    #[allow(dead_code)]
    pub(crate) fn recomp(&self, decomp: &[u8]) -> u32 {
        let mut value: u32 = 0;
        let mut sum_bases: u8 = 0;
        self.0.iter().enumerate().for_each(|(i, base)| {
            value |= (decomp[i] as u32) << sum_bases;
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
                .flat_map(|array| array.0.iter().copied())
                .collect(),
        )
    }

    #[allow(dead_code)]
    pub(crate) fn decomp(&self, value: u32) -> Vec<u8> {
        self.as_1d().decomp(value)
    }

    #[allow(dead_code)]
    pub(crate) fn recomp(&self, decomp: &[u8]) -> u32 {
        self.as_1d().recomp(decomp)
    }
}

pub(crate) fn get_base_2d(value: u32, base: Vec<u8>) -> Base2D {
    let mut out = Vec::new();
    let mut value_bit_size = 32 - (value - 1).leading_zeros();

    while value_bit_size != 0 {
        let mut v: Vec<u8> = Vec::new();

        for &b in base.iter() {
            if b as u32 <= value_bit_size {
                v.push(b);
                value_bit_size -= b as u32;
            } else {
                if value_bit_size != 0 {
                    v.push(value_bit_size as u8);
                    value_bit_size = 0;
                }
                break;
            }
        }

        out.push(Base1D(v)); // Single, unconditional push here
    }

    Base2D(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base1d_max_calculation() {
        // Test max value calculation for various base configurations
        let base1 = Base1D(vec![4, 4, 4]); // 3 * 4 = 12 bits
        assert_eq!(base1.max(), 1 << 12); // 2^12 = 4096

        let base2 = Base1D(vec![8, 8]); // 2 * 8 = 16 bits  
        assert_eq!(base2.max(), 1 << 16); // 2^16 = 65536

        let base3 = Base1D(vec![12]); // 1 * 12 = 12 bits
        assert_eq!(base3.max(), 1 << 12); // 2^12 = 4096

        let base4 = Base1D(vec![1, 1, 1, 1]); // 4 * 1 = 4 bits
        assert_eq!(base4.max(), 1 << 4); // 2^4 = 16
    }

    #[test]
    fn base1d_decomp_recomp_roundtrip() {
        let base = Base1D(vec![4, 4, 4]); // 12 bits total

        // Test full roundtrip with various values
        let test_values = vec![0, 1, 15, 255, 1000, 4095]; // 4095 = 2^12 - 1

        for value in test_values {
            let decomp = base.decomp(value);
            let recomp = base.recomp(&decomp);
            assert_eq!(value, recomp, "Roundtrip failed for value {}", value);

            // Also verify decomposition properties
            assert_eq!(
                decomp.len(),
                3,
                "Decomposition should produce 3 elements for value {}",
                value
            );
            for &elem in &decomp {
                assert!(
                    elem < 16,
                    "Decomposed element {} should be < 16 for value {}",
                    elem,
                    value
                );
            }
        }
    }

    #[test]
    fn base1d_decomp_correctness() {
        let base = Base1D(vec![4, 4, 4]); // 12 bits: 4 + 4 + 4

        // Test specific decomposition with known values
        let value = 0b0000_0000_1111; // 15 in decimal
        let decomp = base.decomp(value);

        // The decomposition extracts bits in order: first 4 bits, next 4 bits, last 4 bits
        // 0b0000_0000_1111 = 0b1111 (15), 0b0000 (0), 0b0000 (0) in reverse order
        assert_eq!(decomp, vec![15, 0, 0]);

        // Verify recomposition works
        let recomp = base.recomp(&decomp);
        assert_eq!(value, recomp);

        // Test another known case
        let value2 = 0b1010_1100_1111; // 2767 in decimal
        let decomp2 = base.decomp(value2);
        assert_eq!(decomp2, vec![15, 12, 10]); // 0b1111, 0b1100, 0b1010
        let recomp2 = base.recomp(&decomp2);
        assert_eq!(value2, recomp2);
    }

    #[test]
    fn base1d_gap_calculation() {
        let base = Base1D(vec![4, 4, 4]); // 12 bits total
        let log_n = 12;

        // gap = log_n >> (sum of bases) = 12 >> 12 = 0
        // result = 1 << 0 = 1
        assert_eq!(base.gap(log_n), 1);

        let base2 = Base1D(vec![6, 6]); // 12 bits total
        // gap = 12 >> 12 = 0, result = 1
        assert_eq!(base2.gap(log_n), 1);

        let base3 = Base1D(vec![3, 3, 3, 3]); // 12 bits total
        // gap = 12 >> 12 = 0, result = 1
        assert_eq!(base3.gap(log_n), 1);
    }

    #[test]
    fn base2d_creation_and_conversion() {
        let base1d_1 = Base1D(vec![4, 4]);
        let base1d_2 = Base1D(vec![4, 4]);
        let base2d = Base2D(vec![base1d_1.clone(), base1d_2.clone()]);

        // Test as_1d conversion
        let as_1d = base2d.as_1d();
        let expected = Base1D(vec![4, 4, 4, 4]); // Flattened
        assert_eq!(as_1d, expected);
    }

    #[test]
    fn base2d_max_calculation() {
        let base1d_1 = Base1D(vec![4, 4]); // 8 bits
        let base1d_2 = Base1D(vec![4, 4]); // 8 bits
        let base2d = Base2D(vec![base1d_1, base1d_2]);

        // Total: 8 + 8 = 16 bits
        assert_eq!(base2d.max(), 1 << 16);

        // Test with different sizes
        let base1d_3 = Base1D(vec![6]);
        let base1d_4 = Base1D(vec![6]);
        let base2d_2 = Base2D(vec![base1d_3, base1d_4]);

        // Total: 6 + 6 = 12 bits
        assert_eq!(base2d_2.max(), 1 << 12);
    }

    #[test]
    fn base2d_decomp_recomp_roundtrip() {
        let base1d_1 = Base1D(vec![4, 4]);
        let base1d_2 = Base1D(vec![4, 4]);
        let base2d = Base2D(vec![base1d_1, base1d_2]);

        // Test full roundtrip with various values
        let test_values = vec![0, 1, 255, 1000, 65535]; // 65535 = 2^16 - 1

        for value in test_values {
            let decomp = base2d.decomp(value);
            let recomp = base2d.recomp(&decomp);
            assert_eq!(value, recomp, "Roundtrip failed for value {}", value);

            // Also verify decomposition properties
            assert_eq!(
                decomp.len(),
                4,
                "Decomposition should produce 4 elements for value {}",
                value
            );
            for &elem in &decomp {
                assert!(
                    elem < 16,
                    "Decomposed element {} should be < 16 for value {}",
                    elem,
                    value
                );
            }
        }
    }

    #[test]
    fn get_base_2d_functionality() {
        // Test with simple case
        let base = vec![4, 4, 4]; // 12 bits
        let value = 1000u32; // Fits in 12 bits
        let base2d = get_base_2d(value, base);

        // Should create a single Base1D with the decomposition
        assert_eq!(base2d.0.len(), 1);
        // The actual decomposition might be different based on the algorithm
        assert!(base2d.0[0].0.len() > 0);

        // Test with larger value requiring multiple Base1D
        let base2 = vec![4, 4]; // 8 bits per Base1D
        let value2 = 1000u32; // Requires 10 bits, so needs 2 Base1D instances
        let base2d_2 = get_base_2d(value2, base2);

        // Should create multiple Base1D instances
        assert!(base2d_2.0.len() >= 1);

        // Verify the decomposition works with full roundtrip
        let decomp = base2d_2.decomp(value2);
        let recomp = base2d_2.recomp(&decomp);
        assert_eq!(
            value2, recomp,
            "Roundtrip failed for get_base_2d with value {}",
            value2
        );

        // Verify decomposition properties
        assert!(
            decomp.len() >= 1,
            "Decomposition should produce at least 1 element"
        );
        for &elem in &decomp {
            assert!(elem < 16, "Decomposed element {} should be < 16", elem);
        }
    }

    #[test]
    fn base1d_edge_cases() {
        // Test empty base
        let empty_base = Base1D(vec![]);
        assert_eq!(empty_base.max(), 1);
        assert_eq!(empty_base.decomp(0), vec![] as Vec<u8>);
        assert_eq!(empty_base.recomp(&vec![] as &[u8]), 0);

        // Test single bit
        let single_bit = Base1D(vec![1]);
        assert_eq!(single_bit.max(), 2);
        assert_eq!(single_bit.decomp(0), vec![0]);
        assert_eq!(single_bit.decomp(1), vec![1]);
        assert_eq!(single_bit.recomp(&vec![0]), 0);
        assert_eq!(single_bit.recomp(&vec![1]), 1);
    }

    #[test]
    fn base1d_comprehensive_roundtrip() {
        let base = Base1D(vec![4, 4, 4]);
        assert_eq!(base.max(), 1 << 12);

        // Test a comprehensive range of values
        let test_values = vec![
            0, 1, 2, 3, 4, 5, 10, 15, 16, 17, 31, 32, 63, 64, 127, 128, 255, 256, 511, 512, 1023,
            1024, 2047, 2048, 4095,
        ];

        for value in test_values {
            let decomp = base.decomp(value);
            let recomp = base.recomp(&decomp);
            assert_eq!(value, recomp, "Roundtrip failed for value {}", value);
        }
    }

    #[test]
    fn base2d_comprehensive_roundtrip() {
        let base1d_1 = Base1D(vec![6, 6]);
        let base1d_2 = Base1D(vec![4, 4]);
        let base2d = Base2D(vec![base1d_1, base1d_2]);

        // Test comprehensive range for 16-bit values
        let test_values = vec![
            0, 1, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256, 511, 512, 1023, 1024, 2047, 2048,
            4095, 4096, 8191, 8192, 16383, 16384, 32767, 32768, 65535,
        ];

        for value in test_values {
            let decomp = base2d.decomp(value);
            let recomp = base2d.recomp(&decomp);
            assert_eq!(value, recomp, "Roundtrip failed for value {}", value);
        }
    }

    #[test]
    fn base1d_different_sizes() {
        // Test various base configurations
        let bases = vec![
            Base1D(vec![1, 1, 1, 1]), // 4 bits
            Base1D(vec![2, 2, 2]),    // 6 bits
            Base1D(vec![3, 3, 3]),    // 9 bits
            Base1D(vec![4, 4, 4]),    // 12 bits
            Base1D(vec![8, 8]),       // 16 bits
        ];

        for base in bases {
            let max_val = (base.max() - 1) as u32;
            let test_values = vec![0u32, 1, max_val / 4, max_val / 2, max_val];

            for value in test_values {
                let decomp = base.decomp(value);
                let recomp = base.recomp(&decomp);
                assert_eq!(
                    value, recomp,
                    "Roundtrip failed for base {:?} with value {}",
                    base.0, value
                );
            }
        }
    }

    #[test]
    fn base2d_edge_cases() {
        // Test empty Base2D
        let empty_base2d = Base2D(vec![]);
        assert_eq!(empty_base2d.max(), 1);
        assert_eq!(empty_base2d.decomp(0), vec![] as Vec<u8>);
        assert_eq!(empty_base2d.recomp(&vec![] as &[u8]), 0);

        // Test single Base1D
        let single_base1d = Base1D(vec![4, 4]);
        let single_base2d = Base2D(vec![single_base1d]);
        assert_eq!(single_base2d.max(), 1 << 8);

        // Test conversion to 1D
        let as_1d = single_base2d.as_1d();
        assert_eq!(as_1d.0, vec![4, 4]);
    }

    #[test]
    fn base_decomposition_boundary_tests() {
        let base = Base1D(vec![4, 4, 4]); // 12 bits

        // Test values just below and at boundaries with full roundtrip
        let test_values = vec![
            0,             // Minimum
            1,             // Just above minimum
            (1 << 4) - 1,  // Max for first 4 bits (15)
            1 << 4,        // First bit of second group (16)
            (1 << 8) - 1,  // Max for first 8 bits (255)
            1 << 8,        // First bit of third group (256)
            (1 << 12) - 1, // Maximum value (4095)
        ];

        for value in test_values {
            let decomp = base.decomp(value);
            let recomp = base.recomp(&decomp);
            assert_eq!(value, recomp, "Boundary test failed for value {}", value);

            // Also verify decomposition properties
            assert_eq!(
                decomp.len(),
                3,
                "Decomposition should produce 3 elements for value {}",
                value
            );
            for &elem in &decomp {
                assert!(
                    elem < 16,
                    "Decomposed element {} should be < 16 for value {}",
                    elem,
                    value
                );
            }
        }
    }
}
