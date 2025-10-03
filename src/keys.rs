use poulpy_backend::cpu_fft64_avx::FFT64Avx;
use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Module, ScratchOwned},
    source::Source,
};
use rand_core::{OsRng, TryRngCore};
use std::collections::HashMap;

use poulpy_core::layouts::{GGLWEAutomorphismKey, GGLWETensorKey, GLWECiphertext, GLWESecret};

#[cfg(test)]
use poulpy_core::layouts::{
    Infos,
    prepared::{GGLWEAutomorphismKeyPrepared, GLWESecretPrepared},
};

use crate::parameters::Parameters;

/// Struct storing the FHE evaluation keys for the read/write on FHE-RAM.
pub struct EvaluationKeys {
    pub(crate) auto_keys: HashMap<i64, GGLWEAutomorphismKey<Vec<u8>>>,
    pub(crate) tensor_key: GGLWETensorKey<Vec<u8>>,
}

/// Generates a new set of [EvaluationKeys] along with the associated secret-key.
pub fn gen_keys(params: &Parameters) -> (GLWESecret<Vec<u8>>, EvaluationKeys) {
    let module: &Module<FFT64Avx> = params.module();
    let basek: usize = params.basek();
    let k_evk: usize = params.k_evk();
    let rows: usize = params.rows_addr();
    let rank: usize = params.rank();
    let digits: usize = params.digits();

    let mut seed_xs: [u8; 32] = [0u8; 32];
    OsRng.try_fill_bytes(&mut seed_xs).unwrap();

    let mut seed_xa: [u8; 32] = [0u8; 32];
    OsRng.try_fill_bytes(&mut seed_xa).unwrap();

    let mut seed_xe: [u8; 32] = [0u8; 32];
    OsRng.try_fill_bytes(&mut seed_xe).unwrap();

    let mut source_xs: Source = Source::new(seed_xs);
    let mut source_xa: Source = Source::new(seed_xa);
    let mut source_xe: Source = Source::new(seed_xe);

    let mut scratch: ScratchOwned<FFT64Avx> = ScratchOwned::alloc(
        GGLWEAutomorphismKey::encrypt_sk_scratch_space(module, basek, k_evk, rank)
            | GGLWEAutomorphismKey::encrypt_sk_scratch_space(module, basek, k_evk, rank)
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
            // params.xe(),
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
        // params.xe(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use poulpy_core::layouts::{prepared::PrepareAlloc, GGLWECiphertext, GLWECiphertext, GLWEPlaintext};
    use poulpy_hal::{
        api::{ScratchOwnedBorrow, VecZnxAutomorphismInplace, VecZnxCopy},
        layouts::ScratchOwned,
        source::Source,
    };

    #[test]
    fn gen_keys_builds_complete_auto_keys_set() {
        let params = Parameters::new();
        let (sk, keys) = gen_keys(&params);

        // Expect one automorphism key per Galois element
        let gal_els = GLWECiphertext::<Vec<u8>>::trace_galois_elements(params.module());
        assert_eq!(keys.auto_keys.len(), gal_els.len());

        // Every required element must exist and match its key parameter `p()`
        for &el in &gal_els {
            let k = keys
                .auto_keys
                .get(&el)
                .expect("missing automorphism key for Galois element");
            assert_eq!(k.p(), el);
        }

        // Encrypt a small plaintext into a GLWE ciphertext with the generated secret key
        let module = params.module();
        let basek = params.basek();
        let k_ct = params.k_ct();
        let k_pt = params.k_pt();
        let rank = params.rank();

        let mut ct_in: GLWECiphertext<Vec<u8>> =
            GLWECiphertext::alloc(module.n(), basek, k_ct, rank);
        let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(module.n(), basek, k_pt);

        let pt_data: Vec<i64> = (0..module.n())
            .map(|i| (i % k_pt as usize) as i64)
            .collect();
        pt.data
            .encode_vec_i64(basek, 0, k_pt, &pt_data, u8::BITS as usize);

        let mut scratch: ScratchOwned<FFT64Avx> = ScratchOwned::alloc(
            GGLWEAutomorphismKey::encrypt_sk_scratch_space(module, basek, k_ct, rank)
            | GLWECiphertext::encrypt_sk_scratch_space(module, basek, k_ct)
        );
        let sk_prepared: GLWESecretPrepared<Vec<u8>, FFT64Avx> =
            sk.prepare_alloc(module, scratch.borrow());
        let mut source_xa: Source = Source::new([3u8; 32]);
        let mut source_xe: Source = Source::new([4u8; 32]);
        ct_in.encrypt_sk(
            module,
            &pt,
            &sk_prepared,
            &mut source_xa,
            &mut source_xe,
            scratch.borrow(),
        );

        // Test all automorphisms
        for &gal_el in &gal_els {
            let auto_key = keys
                .auto_keys
                .get(&gal_el)
                .expect("automorphism key not found for Galois element");
            let auto_key_prepared: GGLWEAutomorphismKeyPrepared<Vec<u8>, FFT64Avx> =
                auto_key.prepare_alloc(module, scratch.borrow());

            let mut ct_out: GLWECiphertext<Vec<u8>> =
                GLWECiphertext::alloc(module.n(), basek, k_ct, rank);
            ct_out.automorphism(module, &ct_in, &auto_key_prepared, scratch.borrow());

            let mut pt_out: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(module.n(), basek, k_pt);
            ct_out.decrypt(module, &mut pt_out, &sk_prepared, scratch.borrow());

            // Apply the same automorphism to the original plaintext
            let mut pt_expected: GLWEPlaintext<Vec<u8>> =
                GLWEPlaintext::alloc(module.n(), basek, k_pt);
            module.vec_znx_copy(&mut pt_expected.data, 0, &pt.data, 0);
            module.vec_znx_automorphism_inplace(gal_el, &mut pt_expected.data, 0, scratch.borrow());

            // Verify the decrypted result matches the expected automorphism of the plaintext
            assert_eq!(
                pt_expected.data, pt_out.data,
                "Automorphism failed for Galois element {}",
                gal_el
            );

            // Structural invariants
            assert_eq!(ct_out.basek(), ct_in.basek());
            assert_eq!(ct_out.k(), ct_in.k());
            assert_eq!(ct_out.n(), ct_in.n());

            // For non-identity elements, the ciphertext should change
            if gal_el != 1 {
                assert_ne!(
                    ct_out, ct_in,
                    "Ciphertext unchanged for non-identity Galois element {}",
                    gal_el
                );
            } else {
                // For identity element, ciphertext should remain the same
                assert_eq!(
                    ct_out, ct_in,
                    "Ciphertext changed for identity Galois element"
                );
            }
        }
    }
}
