use std::collections::HashMap;

use poulpy_backend::FFT64Spqlios;
use poulpy_core::layouts::{GGLWEAutomorphismKey, GGLWETensorKey, GLWECiphertext, GLWESecret};
use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Module, ScratchOwned},
    source::Source,
};
use rand_core::{OsRng, TryRngCore};

use crate::parameters::Parameters;

/// Struct storing the FHE evaluation keys for the read/write on FHE-RAM.
pub struct EvaluationKeys {
    pub(crate) auto_keys: HashMap<i64, GGLWEAutomorphismKey<Vec<u8>>>,
    pub(crate) tensor_key: GGLWETensorKey<Vec<u8>>,
}

/// Generates a new set of [EvaluationKeys] along with the associated secret-key.
pub fn gen_keys(params: &Parameters) -> (GLWESecret<Vec<u8>>, EvaluationKeys) {
    let module: &Module<FFT64Spqlios> = params.module();
    let basek: usize = params.basek();
    let k_evk: usize = params.k_evk();
    let rows: usize = params.rows_addr();
    let rank: usize = params.rank();
    let digits: usize = params.digits();

    let mut root = [0u8; 32];
    OsRng.try_fill_bytes(&mut root).unwrap();

    let mut source: Source = Source::new(root);

    let seed_xs = source.new_seed();
    let mut source_xs = Source::new(seed_xs);

    let seed_xa = source.new_seed();
    let mut source_xa = Source::new(seed_xa);

    let seed_xe = source.new_seed();
    let mut source_xe = Source::new(seed_xe);

    let mut scratch: ScratchOwned<FFT64Spqlios> = ScratchOwned::alloc(
        GGLWEAutomorphismKey::encrypt_sk_scratch_space(module, basek, k_evk, rank)
            | GGLWETensorKey::encrypt_sk_scratch_space(module, basek, k_evk, rank),
    );

    let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc(module.n(), params.rank());
    sk.fill_ternary_prob(params.xs(), &mut source_xs);

    let gal_els: Vec<i64> = GLWECiphertext::trace_galois_elements(module);
    let auto_keys = HashMap::from_iter(gal_els.iter().map(|gal_el| {
        let mut key: GGLWEAutomorphismKey<Vec<u8>> =
            GGLWEAutomorphismKey::alloc(module.n(), basek, k_evk, rows, digits, rank);
        key.encrypt_sk(
            module,
            *gal_el,
            &sk,
            &mut source_xa,
            &mut source_xe,
            scratch.borrow(),
        );
        (*gal_el, key)
    }));

    let mut tensor_key = GGLWETensorKey::alloc(module.n(), basek, k_evk, rows, digits, rank);
    tensor_key.encrypt_sk(
        module,
        &sk,
        &mut source_xa,
        &mut source_xe,
        scratch.borrow(),
    );

    (
        sk,
        EvaluationKeys {
            auto_keys,
            tensor_key,
        },
    )
}
