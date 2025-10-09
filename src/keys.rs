use poulpy_hal::{
    api::{
        ScratchAvailable, ScratchOwnedAlloc, ScratchOwnedBorrow, SvpApplyDftToDft,
        SvpApplyDftToDftInplace, SvpPPolAllocBytes, SvpPrepare, TakeScalarZnx, TakeVecZnx,
        TakeVecZnxBig, TakeVecZnxDft, VecZnxAddInplace, VecZnxAddNormal, VecZnxAddScalarInplace,
        VecZnxAutomorphism, VecZnxBigAllocBytes, VecZnxBigNormalize, VecZnxDftAllocBytes,
        VecZnxDftApply, VecZnxFillUniform, VecZnxIdftApplyConsume, VecZnxIdftApplyTmpA,
        VecZnxNormalize, VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes, VecZnxSub,
        VecZnxSubInplace, VecZnxSwitchRing, VmpPMatAlloc, VmpPrepare,
    },
    layouts::{Backend, Data, DataRef, Module, Scratch, ScratchOwned},
    source::Source,
};
use std::collections::HashMap;

use poulpy_core::{
    TakeGLWESecretPrepared,
    layouts::{
        GGLWEAutomorphismKey, GGLWECiphertextLayout, GGLWETensorKey, GLWECiphertext, GLWESecret,
        prepared::{GGLWEAutomorphismKeyPrepared, GGLWETensorKeyPrepared, PrepareAlloc},
    },
};

use crate::parameters::Parameters;

/// Struct storing the FHE evaluation keys for the read/write on FHE-RAM.
pub struct EvaluationKeys<D: Data> {
    atk_glwe: HashMap<i64, GGLWEAutomorphismKey<D>>,
    atk_ggsw_inv: GGLWEAutomorphismKey<D>,
    tsk_ggsw_inv: GGLWETensorKey<D>,
}

pub struct EvaluationKeysPrepared<D: Data, B: Backend> {
    pub(crate) atk_glwe: HashMap<i64, GGLWEAutomorphismKeyPrepared<D, B>>,
    pub(crate) atk_ggsw_inv: GGLWEAutomorphismKeyPrepared<D, B>,
    pub(crate) tsk_ggsw_inv: GGLWETensorKeyPrepared<D, B>,
}

impl<B: Backend, DR: DataRef> PrepareAlloc<B, EvaluationKeysPrepared<Vec<u8>, B>>
    for EvaluationKeys<DR>
where
    Module<B>: VmpPMatAlloc<B> + VmpPrepare<B>,
{
    fn prepare_alloc(
        &self,
        module: &Module<B>,
        scratch: &mut Scratch<B>,
    ) -> EvaluationKeysPrepared<Vec<u8>, B> {
        EvaluationKeysPrepared {
            atk_glwe: HashMap::from_iter(self.atk_glwe.iter().map(|(gal_el, key)| {
                let key_prepared = key.prepare_alloc(module, scratch);
                (*gal_el, key_prepared)
            })),
            atk_ggsw_inv: self.atk_ggsw_inv.prepare_alloc(module, scratch),
            tsk_ggsw_inv: self.tsk_ggsw_inv.prepare_alloc(module, scratch),
        }
    }
}

impl EvaluationKeys<Vec<u8>> {
    /// Constructor for EvaluationKeys
    pub fn new(
        atk_glwe: HashMap<i64, GGLWEAutomorphismKey<Vec<u8>>>,
        atk_ggsw_inv: GGLWEAutomorphismKey<Vec<u8>>,
        tsk_ggsw_inv: GGLWETensorKey<Vec<u8>>,
    ) -> Self {
        Self {
            atk_glwe,
            atk_ggsw_inv,
            tsk_ggsw_inv,
        }
    }

    /// Getter for auto_keys at glwe level
    pub fn atk_glwe(&self) -> &HashMap<i64, GGLWEAutomorphismKey<Vec<u8>>> {
        &self.atk_glwe
    }

    /// Mutable getter for auto_keys at glwe level
    pub fn atk_glwe_mut(&mut self) -> &mut HashMap<i64, GGLWEAutomorphismKey<Vec<u8>>> {
        &mut self.atk_glwe
    }

    /// Setter for auto_keys at glwe level
    pub fn set_atk_glwe(&mut self, atk_glwe: HashMap<i64, GGLWEAutomorphismKey<Vec<u8>>>) {
        self.atk_glwe = atk_glwe;
    }

    /// Getter for tensor_key at ggsw level
    pub fn tsk_ggsw_inv(&self) -> &GGLWETensorKey<Vec<u8>> {
        &self.tsk_ggsw_inv
    }

    /// Mutable getter for tensor_key at ggsw level
    pub fn tsk_ggsw_inv_mut(&mut self) -> &mut GGLWETensorKey<Vec<u8>> {
        &mut self.tsk_ggsw_inv
    }

    /// Setter for tensor_key at ggsw level
    pub fn set_tsk_ggsw_inv(&mut self, tsk_ggsw_inv: GGLWETensorKey<Vec<u8>>) {
        self.tsk_ggsw_inv = tsk_ggsw_inv;
    }

    /// Getter for auto_key(-1) at ggsw level
    pub fn atk_ggsw_inv(&self) -> &GGLWEAutomorphismKey<Vec<u8>> {
        &self.atk_ggsw_inv
    }

    /// Mutable getter for auto_key(-1) at ggsw level
    pub fn atk_ggsw_inv_mut(&mut self) -> &mut GGLWEAutomorphismKey<Vec<u8>> {
        &mut self.atk_ggsw_inv
    }

    /// Setter for auto_key(-1) at ggsw level
    pub fn set_atk_ggsw_inv(&mut self, atk_ggsw_inv: GGLWEAutomorphismKey<Vec<u8>>) {
        self.atk_ggsw_inv = atk_ggsw_inv;
    }
}

impl EvaluationKeys<Vec<u8>> {
    pub fn encrypt_sk<S, B: Backend>(
        params: &Parameters<B>,
        sk: &GLWESecret<S>,
        source_xa: &mut Source,
        source_xe: &mut Source,
    ) -> EvaluationKeys<Vec<u8>>
    where
        S: DataRef,
        ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
        Module<B>: SvpPPolAllocBytes
            + VecZnxNormalizeTmpBytes
            + VecZnxDftAllocBytes
            + VecZnxNormalizeTmpBytes
            + VecZnxBigAllocBytes,
        Module<B>: VecZnxAddScalarInplace
            + VecZnxDftAllocBytes
            + VecZnxBigNormalize<B>
            + VecZnxDftApply<B>
            + SvpApplyDftToDftInplace<B>
            + VecZnxIdftApplyConsume<B>
            + VecZnxNormalizeTmpBytes
            + VecZnxFillUniform
            + VecZnxSubInplace
            + VecZnxAddInplace
            + VecZnxNormalizeInplace<B>
            + VecZnxAddNormal
            + VecZnxNormalize<B>
            + VecZnxSub
            + SvpPrepare<B>
            + VecZnxSwitchRing
            + SvpPPolAllocBytes
            + VecZnxAutomorphism,
        Scratch<B>: TakeVecZnxDft<B>
            + ScratchAvailable
            + TakeVecZnx
            + TakeScalarZnx
            + TakeGLWESecretPrepared<B>,
        Module<B>: SvpApplyDftToDft<B> + VecZnxIdftApplyTmpA<B>,
        Scratch<B>: TakeVecZnxBig<B>,
    {
        let module: &Module<B> = params.module();

        let evk_glwe_infos: GGLWECiphertextLayout = params.evk_glwe_infos();
        let evk_ggsw_infos: GGLWECiphertextLayout = params.evk_ggsw_infos();

        let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(
            GGLWEAutomorphismKey::encrypt_sk_scratch_space(module, &evk_glwe_infos)
                | GGLWEAutomorphismKey::encrypt_sk_scratch_space(module, &evk_ggsw_infos)
                | GGLWETensorKey::encrypt_sk_scratch_space(module, &evk_ggsw_infos),
        );

        let gal_els: Vec<i64> = GLWECiphertext::trace_galois_elements(module);
        let atk_glwe: HashMap<i64, GGLWEAutomorphismKey<Vec<u8>>> =
            HashMap::from_iter(gal_els.iter().map(|gal_el| {
                let mut key: GGLWEAutomorphismKey<Vec<u8>> =
                    GGLWEAutomorphismKey::alloc(&evk_glwe_infos);
                key.encrypt_sk(module, *gal_el, sk, source_xa, source_xe, scratch.borrow());
                (*gal_el, key)
            }));

        let mut tsk_ggsw_inv: GGLWETensorKey<Vec<u8>> = GGLWETensorKey::alloc(&evk_ggsw_infos);
        tsk_ggsw_inv.encrypt_sk(module, sk, source_xa, source_xe, scratch.borrow());

        let mut atk_ggsw_inv: GGLWEAutomorphismKey<Vec<u8>> =
            GGLWEAutomorphismKey::alloc(&evk_ggsw_infos);
        atk_ggsw_inv.encrypt_sk(module, -1, sk, source_xa, source_xe, scratch.borrow());

        EvaluationKeys {
            atk_glwe,
            atk_ggsw_inv,
            tsk_ggsw_inv,
        }
    }
}
