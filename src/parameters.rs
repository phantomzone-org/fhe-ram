use backend::{FFT64, Module};

const LOG_N: usize = 10;
const BASEK: usize = 17;
const RANK: usize = 3;
const K_PT: usize = (u8::BITS as usize) + 1;
const K_CT: usize = BASEK * 3;
const K_ADDR: usize = BASEK * 4;
const K_EVK: usize = BASEK * 5;
const XS: f64 = 0.5;
const XE: f64 = 3.2;
pub const DECOMP_N: [u8; 3] = [4, 3, 3];

pub const MAX_ADDR: usize = 65536;

pub struct Parameters {
    module: Module<FFT64>,
    basek: usize,
    rank: usize,
    k_pt: usize,
    k_ct: usize,
    k_addr: usize,
    k_evk: usize,
    xs: f64,
    xe: f64,
    max_addr: usize,
    decomp_n: Vec<u8>,
}

impl Parameters {
    pub fn new() -> Self {
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

    pub fn size_ct(&self) -> usize {
        (self.k_ct() + self.basek() - 1) / self.basek()
    }

    pub(crate) fn k_evk(&self) -> usize {
        self.k_evk
    }

    pub(crate) fn xs(&self) -> f64 {
        self.xs
    }

    pub(crate) fn size_addr(&self) -> usize {
        (self.k_addr() + self.basek() - 1) / self.basek()
    }

    pub(crate) fn size_evk(&self) -> usize {
        (self.k_evk() + self.basek() - 1) / self.basek()
    }

    pub(crate) fn rows_ct(&self) -> usize {
        self.size_ct()
    }

    pub(crate) fn rows_addr(&self) -> usize {
        self.size_addr()
    }

    pub(crate) fn decomp_n(&self) -> Vec<u8> {
        self.decomp_n.clone()
    }
}
