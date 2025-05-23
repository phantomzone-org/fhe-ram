use backend::{FFT64, Module};

const LOG_N: usize = 10;
const RANK: usize = 2;
const K_CT: usize = 54;
const K_PT: usize = 8;
const K_EVK: usize = 64;
const BASEK: usize = 18;
const ROWS: usize = (K_CT + BASEK - 1) / BASEK;
const EVK_SIZE: usize = (K_EVK + BASEK - 1) / BASEK;
const XS: f64 = 0.5;
const XE: f64 = 3.2;
pub const DECOMP_N: [u8; 3] = [4, 3, 3];

pub struct Parameters {
    pub(crate) module: Module<FFT64>,
    pub(crate) basek: usize,
    pub(crate) rows: usize,
    pub(crate) rank: usize,
    pub(crate) k_ct: usize,
    pub(crate) k_pt: usize,
    pub(crate) k_evk: usize,
    pub(crate) size_evk: usize,
    pub(crate) xs: f64,
    pub(crate) xe: f64,
    pub(crate) decomp_n: Vec<u8>,
}

impl Parameters {
    pub fn new() -> Self {
        Self {
            module: Module::<FFT64>::new(1 << LOG_N),
            basek: BASEK,
            rows: ROWS,
            rank: RANK,
            k_ct: K_CT,
            k_pt: K_PT,
            k_evk: K_CT,
            size_evk: EVK_SIZE,
            xs: XS,
            xe: XE,
            decomp_n: DECOMP_N.to_vec(),
        }
    }
}
