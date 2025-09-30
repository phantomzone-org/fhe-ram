use poulpy_backend::cpu_fft64_avx::FFT64Avx;
use poulpy_hal::{
    api::ModuleNew,
    layouts::Module,
};

const LOG_N: usize = 12;
const BASEK: usize = 17;
const RANK: usize = 1;
const K_PT: usize = u8::BITS as usize;
const K_CT: usize = BASEK * 3;
const K_ADDR: usize = BASEK * 4;
const K_EVK: usize = BASEK * 5;
const XS: f64 = 0.5;
const XE: f64 = 3.2;
pub const DECOMP_N: [u8; 1] = [12];
const WORDSIZE: usize = 4;
const MAX_ADDR: usize = 1 << 18;
const DIGITS: usize = 1;

pub struct Parameters {
    module: Module<FFT64Avx>, // FFT/NTT tables.
    basek: usize,          // Torus 2^{-k} decomposition.
    digits: usize,         // Digits of GGLWE/GGSW product
    rank: usize,           // GLWE/GGLWE/GGSW rank.
    k_pt: usize,           // Ram plaintext (GLWE) Torus precision.
    k_ct: usize,           // Ram ciphertext (GLWE) Torus precision.
    k_addr: usize,         // Ram address (GGSW) Torus precision.
    k_evk: usize,          // Ram evaluation keys (GGLWE) Torus precision
    xs: f64,               // Secret-key distribution.
    xe: f64,               // Noise standard deviation.
    max_addr: usize,       // Maximum supported address.
    decomp_n: Vec<u8>,     // Digit decomposition of N.
    word_size: usize,      // Digit decomposition of a Ram word.
}

impl Parameters {
    pub fn new() -> Self {
        assert!(DECOMP_N.iter().sum::<u8>() == LOG_N as u8);

        Self {
            module: Module::<FFT64Avx>::new(1 << LOG_N),
            basek: BASEK,
            digits: DIGITS,
            rank: RANK,
            k_ct: K_CT,
            k_pt: K_PT,
            k_addr: K_ADDR,
            k_evk: K_EVK,
            xs: XS,
            xe: XE,
            max_addr: MAX_ADDR,
            decomp_n: DECOMP_N.to_vec(),
            word_size: WORDSIZE,
        }
    }

    pub fn max_addr(&self) -> usize {
        self.max_addr
    }

    pub fn module(&self) -> &Module<FFT64Avx> {
        &self.module
    }

    pub fn basek(&self) -> usize {
        self.basek
    }

    pub fn k_ct(&self) -> usize {
        self.k_ct
    }

    pub fn k_pt(&self) -> usize {
        self.k_pt
    }

    pub(crate) fn k_addr(&self) -> usize {
        self.k_addr
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn xe(&self) -> f64 {
        self.xe
    }

    pub fn word_size(&self) -> usize {
        self.word_size
    }

    pub(crate) fn digits(&self) -> usize {
        self.digits
    }

    pub(crate) fn k_evk(&self) -> usize {
        self.k_evk
    }

    pub(crate) fn xs(&self) -> f64 {
        self.xs
    }

    pub(crate) fn rows_ct(&self) -> usize {
        (self.k_ct() + self.basek() - 1) / self.basek()
    }

    pub(crate) fn rows_addr(&self) -> usize {
        (self.k_addr() + self.basek() - 1) / self.basek()
    }

    pub(crate) fn decomp_n(&self) -> Vec<u8> {
        self.decomp_n.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_sets_expected_defaults() {
        let p = Parameters::new();

        // Public accessors
        assert_eq!(p.basek(), BASEK);
        assert_eq!(p.k_ct(), K_CT);
        assert_eq!(p.k_pt(), K_PT);
        assert_eq!(p.rank(), RANK);
        assert_eq!(p.xe(), XE);
        assert_eq!(p.word_size(), WORDSIZE);
        assert_eq!(p.max_addr(), MAX_ADDR);

        // Module size
        assert_eq!(p.module().n(), 1 << LOG_N);

        // Crate-visible accessors/fields
        assert_eq!(p.k_addr(), K_ADDR);
        assert_eq!(p.k_evk(), K_EVK);
        assert_eq!(p.digits(), DIGITS);
        assert_eq!(p.xs(), XS);

        // Decomposition of N
        assert_eq!(p.decomp_n(), DECOMP_N.to_vec());
        assert_eq!(p.decomp_n().iter().copied().sum::<u8>(), LOG_N as u8);
    }

    #[test]
    fn rows_are_computed_correctly() {
        let p = Parameters::new();

        let expected_rows_ct = (K_CT + BASEK - 1) / BASEK;
        let expected_rows_addr = (K_ADDR + BASEK - 1) / BASEK;

        assert_eq!(p.rows_ct(), expected_rows_ct);
        assert_eq!(p.rows_addr(), expected_rows_addr);
    }
}
