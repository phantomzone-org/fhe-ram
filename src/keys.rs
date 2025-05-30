use core::{AutomorphismKey, GLWECiphertext, GLWESecret, TensorKey};
use std::collections::HashMap;

use backend::{FFT64, Module, ScratchOwned};
use sampling::source::{Source, new_seed};

use crate::parameters::Parameters;

/// Struct storing the FHE evaluation keys for the read/write on FHE-RAM.
pub struct EvaluationKeys {
    pub(crate) auto_keys: HashMap<i64, AutomorphismKey<Vec<u8>, FFT64>>,
    pub(crate) tensor_key: TensorKey<Vec<u8>, FFT64>,
}

/// Generates a new set of [EvaluationKeys] along with the associated secret-key.
pub fn gen_keys(params: &Parameters) -> (GLWESecret<Vec<u8>, FFT64>, EvaluationKeys) {
    let module: &Module<FFT64> = &params.module();
    let basek: usize = params.basek();
    let k_evk: usize = params.k_evk();
    let rows: usize = params.rows_addr();
    let rank: usize = params.rank();

    let mut source_1: Source = Source::new(new_seed());
    let mut source_2: Source = Source::new(new_seed());

    let mut scratch: ScratchOwned = ScratchOwned::new(
        AutomorphismKey::generate_from_sk_scratch_space(module, basek, k_evk, rank)
            | AutomorphismKey::generate_from_sk_scratch_space(module, basek, k_evk, rank)
            | TensorKey::generate_from_sk_scratch_space(module, basek, k_evk, rank),
    );

    let mut sk: GLWESecret<Vec<u8>, FFT64> = GLWESecret::alloc(module, params.rank());
    sk.fill_ternary_prob(&module, params.xs(), &mut source_1);

    let gal_els: Vec<i64> = GLWECiphertext::trace_galois_elements(&module);

    let mut auto_keys: HashMap<i64, AutomorphismKey<Vec<u8>, FFT64>> = HashMap::new();
    gal_els.iter().for_each(|gal_el| {
        let mut key: AutomorphismKey<Vec<u8>, FFT64> =
            AutomorphismKey::alloc(&module, basek, k_evk, rows, rank);
        key.generate_from_sk(
            &module,
            *gal_el,
            &sk,
            &mut source_1,
            &mut source_2,
            params.xe(),
            scratch.borrow(),
        );
        auto_keys.insert(*gal_el, key);
    });

    let mut tensor_key = TensorKey::alloc(module, basek, k_evk, rows, rank);
    tensor_key.generate_from_sk(
        module,
        &sk,
        &mut source_1,
        &mut source_2,
        params.xe(),
        scratch.borrow(),
    );

    (sk, EvaluationKeys {
        auto_keys,
        tensor_key,
    })
}
