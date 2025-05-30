use backend::{FFT64, Module};

const LOG_N: usize = 12;
const BASEK: usize = 17;
const RANK: usize = 1;
const K_PT: usize = (u8::BITS as usize) + 1;
const K_CT: usize = BASEK * 3;
const K_ADDR: usize = BASEK * 4;
const K_EVK: usize = BASEK * 5;
const XS: f64 = 0.5;
const XE: f64 = 3.2;
pub const DECOMP_N: [u8; 2] = [6, 6];

pub const RAM_CHUNKS: usize = 4;
pub const MAX_ADDR: usize = 1 << 18;

pub struct Parameters {
    module: Module<FFT64>, // FFT/NTT tables.
    basek: usize,          // Torus 2^{-k} decomposition.
    rank: usize,           // GLWE/GGLWE/GGSW rank.
    k_pt: usize,           // Ram plaintext (GLWE) Torus precision.
    k_ct: usize,           // Ram ciphertext (GLWE) Torus precision.
    k_addr: usize,         // Ram address (GGSW) Torus precision.
    k_evk: usize,          // Ram evaluation keys (GGLWE) Torus precision
    xs: f64,               // Secret-key distribution.
    xe: f64,               // Noise standard deviation.
    max_addr: usize,       // Maximum supported address.
    decomp_n: Vec<u8>,     // Digit decomposition of N.
    ram_chunks: usize,     // Digit decomposition of a Ram word.
}

impl Parameters {
    pub fn new() -> Self {
        assert!(DECOMP_N.iter().sum::<u8>() == LOG_N as u8);

        Self {
            module: Module::<FFT64>::new(1 << LOG_N),
            basek: BASEK,
            rank: RANK,
            k_ct: K_CT,
            k_pt: K_PT,
            k_addr: K_ADDR,
            k_evk: K_EVK,
            xs: XS,
            xe: XE,
            max_addr: MAX_ADDR,
            decomp_n: DECOMP_N.to_vec(),
            ram_chunks: RAM_CHUNKS,
        }
    }

    pub fn max_addr(&self) -> usize {
        self.max_addr
    }

    pub fn module(&self) -> &Module<FFT64> {
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

    pub(crate) fn ram_chunks(&self) -> usize {
        self.ram_chunks
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
