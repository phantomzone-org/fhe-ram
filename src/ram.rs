use std::collections::HashMap;

use itertools::izip;

use poulpy_core::{
    GLWEExternalProductInplace, GLWEOperations, GLWEPacker, TakeGGSW, TakeGGSWPrepared,
    TakeGGSWPreparedSlice, TakeGLWECt,
    layouts::{
        GGLWEAutomorphismKey, GGLWECiphertextLayout, GGSWCiphertext, GGSWCiphertextLayout,
        GGSWInfos, GLWECiphertext, GLWECiphertextLayout, GLWEPlaintext, GLWESecret, LWEInfos,
        prepared::{
            GGLWEAutomorphismKeyPrepared, GGLWETensorKeyPrepared, GGSWCiphertextPrepared,
            GLWESecretPrepared, Prepare, PrepareAlloc,
        },
    },
};
use poulpy_hal::{
    api::{
        ModuleNew, ScratchAvailable, ScratchOwnedAlloc, ScratchOwnedBorrow,
        SvpApplyDftToDftInplace, TakeScalarZnx, TakeVecZnx, TakeVecZnxBig, TakeVecZnxDft,
        VecZnxAddInplace, VecZnxAddNormal, VecZnxAutomorphismInplace, VecZnxBigAddInplace,
        VecZnxBigAddSmallInplace, VecZnxBigAllocBytes, VecZnxBigAutomorphismInplace,
        VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxBigSubSmallNegateInplace, VecZnxCopy,
        VecZnxDftAddInplace, VecZnxDftAllocBytes, VecZnxDftApply, VecZnxDftCopy, VecZnxFillUniform,
        VecZnxIdftApplyConsume, VecZnxIdftApplyTmpA, VecZnxNegateInplace, VecZnxNormalize,
        VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes, VecZnxRotate, VecZnxRotateInplace,
        VecZnxRshInplace, VecZnxSub, VecZnxSubInplace, VmpApplyDftToDft, VmpApplyDftToDftAdd,
        VmpApplyDftToDftTmpBytes, VmpPMatAlloc, VmpPMatAllocBytes, VmpPrepare, VmpPrepareTmpBytes,
    },
    layouts::{Backend, DataRef, Module, Scratch, ScratchOwned},
    source::Source,
};

use crate::{
    Coordinate, CoordinatePrepared, TakeCoordinatePrepared, address::Address,
    keys::EvaluationKeysPrepared, parameters::Parameters, reverse_bits_msb,
};

/// [Ram] core implementation of the FHE-RAM.
pub struct Ram<B: Backend> {
    params: Parameters<B>,
    subrams: Vec<SubRam>,
    scratch: ScratchOwned<B>,
}

impl<B: Backend> Default for Ram<B>
where
    Module<B>: ModuleNew<B>,
    ScratchOwned<B>: ScratchOwnedAlloc<B>,
    Module<B>: VecZnxNormalizeTmpBytes + VecZnxDftAllocBytes,
    Module<B>: VecZnxDftAllocBytes
        + VmpApplyDftToDftTmpBytes
        + VecZnxBigNormalizeTmpBytes
        + VecZnxNormalizeTmpBytes
        + VecZnxBigAllocBytes
        + VmpPrepareTmpBytes
        + VmpPMatAllocBytes,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> Ram<B>
where
    Module<B>: ModuleNew<B>,
    ScratchOwned<B>: ScratchOwnedAlloc<B>,
    Module<B>: VecZnxNormalizeTmpBytes + VecZnxDftAllocBytes + VecZnxBigAllocBytes,
    Module<B>: VecZnxDftAllocBytes
        + VmpApplyDftToDftTmpBytes
        + VecZnxBigNormalizeTmpBytes
        + VecZnxNormalizeTmpBytes
        + VmpPrepareTmpBytes
        + VmpPMatAllocBytes,
{
    /// Instantiates a new [Ram].
    pub fn new() -> Self {
        let params: Parameters<B> = Parameters::new();
        let scratch: ScratchOwned<B> = ScratchOwned::alloc(Self::scratch_bytes(&params));
        Self {
            subrams: (0..params.word_size())
                .map(|_| SubRam::alloc(&params))
                .collect(),
            params,
            scratch,
        }
    }
}

impl<B: Backend> Ram<B> {
    /// Scratch space size required by the [Ram].
    pub(crate) fn scratch_bytes(params: &Parameters<B>) -> usize
    where
        Module<B>: VecZnxNormalizeTmpBytes + VecZnxDftAllocBytes + VecZnxBigAllocBytes,
        Module<B>: VecZnxDftAllocBytes
            + VmpApplyDftToDftTmpBytes
            + VecZnxBigNormalizeTmpBytes
            + VecZnxNormalizeTmpBytes
            + VmpPrepareTmpBytes
            + VmpPMatAllocBytes,
    {
        let module: &Module<B> = params.module();
        let glwe_infos: GLWECiphertextLayout = params.glwe_ct_infos();
        let ggsw_infos: GGSWCiphertextLayout = params.ggsw_infos();
        let evk_glwe_infos: GGLWECiphertextLayout = params.evk_glwe_infos();

        let enc_sk: usize = GLWECiphertext::encrypt_sk_scratch_space(module, &glwe_infos);

        // Read
        let coordinate_product: usize = Coordinate::product_scratch_space(params);
        let packing: usize = GLWEPacker::scratch_space(module, &glwe_infos, &evk_glwe_infos);
        let trace: usize =
            GLWECiphertext::trace_inplace_scratch_space(module, &glwe_infos, &evk_glwe_infos);
        let ct: usize = GLWECiphertext::alloc_bytes(&glwe_infos);
        let read: usize = coordinate_product.max(trace).max(packing);

        // Write
        let inv_addr: usize =
            CoordinatePrepared::alloc_bytes(module, &ggsw_infos, params.base2d().max_len());
        let prepare_inv: usize = Coordinate::prepare_inv_scratch_space(params);
        let write_first_step: usize = ct + trace;
        let write_mit_step: usize = coordinate_product.max(ct + trace);
        let write_end_step: usize = coordinate_product;
        let write: usize =
            write_first_step.max(inv_addr + (prepare_inv.max(write_mit_step).max(write_end_step)));

        enc_sk.max(read).max(write)
    }

    /// Initialize the FHE-[Ram] with provided values (encrypted inder the provided secret).
    pub fn encrypt_sk(&mut self, data: &[u8], sk: &GLWESecret<Vec<u8>>)
    where
        ScratchOwned<B>: ScratchOwnedBorrow<B>,
        GLWESecret<Vec<u8>>: PrepareAlloc<B, GLWESecretPrepared<Vec<u8>, B>>,
        Module<B>: VecZnxDftAllocBytes
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
            + VecZnxSub,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable + TakeVecZnx,
    {
        let params: &Parameters<B> = &self.params;
        let max_addr: usize = params.max_addr();
        let ram_chunks: usize = params.word_size();

        assert!(
            data.len().is_multiple_of(ram_chunks),
            "invalid data: data.len()%ram_chunks={} != 0",
            data.len().is_multiple_of(ram_chunks),
        );

        assert!(
            data.len() / ram_chunks == max_addr,
            "invalid data: data.len()/ram_chunks={} != max_addr={}",
            data.len() / ram_chunks,
            max_addr
        );

        let scratch: &mut Scratch<B> = self.scratch.borrow();

        let mut data_split: Vec<u8> = vec![0u8; max_addr];

        for i in 0..ram_chunks {
            for (j, x) in data_split.iter_mut().enumerate() {
                *x = data[j * ram_chunks + i];
            }
            self.subrams[i].encrypt_sk(params, &data_split, sk, scratch);
        }
    }

    /// Simple read from the [Ram] at the provided encrypted address.
    /// Returns a vector of [GLWECiphertext], where each ciphertext stores
    /// Enc(m_i) where is the i-th digit of the word-size such that m = m_0 | m-1 | ...
    pub fn read<D: DataRef, DA: DataRef>(
        &mut self,
        address: &Address<DA>,
        keys: &EvaluationKeysPrepared<D, B>,
    ) -> Vec<GLWECiphertext<Vec<u8>>>
    where
        ScratchOwned<B>: ScratchOwnedBorrow<B>,
        GGLWEAutomorphismKey<Vec<u8>>: PrepareAlloc<B, GGLWEAutomorphismKeyPrepared<Vec<u8>, B>>,
        Module<B>: VecZnxDftAllocBytes
            + VmpApplyDftToDftTmpBytes
            + VecZnxBigNormalizeTmpBytes
            + VmpApplyDftToDft<B>
            + VmpApplyDftToDftAdd<B>
            + VecZnxDftApply<B>
            + VecZnxIdftApplyConsume<B>
            + VecZnxBigAddSmallInplace<B>
            + VecZnxBigNormalize<B>
            + VecZnxCopy
            + VecZnxRotateInplace<B>
            + VecZnxSub
            + VecZnxNegateInplace
            + VecZnxRshInplace<B>
            + VecZnxAddInplace
            + VecZnxNormalizeInplace<B>
            + VecZnxSubInplace
            + VecZnxRotate
            + VecZnxAutomorphismInplace<B>
            + VecZnxBigSubSmallNegateInplace<B>
            + VecZnxBigAutomorphismInplace<B>
            + VecZnxNormalize<B>
            + VecZnxNormalizeTmpBytes,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable + TakeVecZnx,
        Scratch<B>: TakeScalarZnx + TakeGGSWPreparedSlice<B>,
        GGSWCiphertext<Vec<u8>>: PrepareAlloc<B, GGSWCiphertextPrepared<Vec<u8>, B>>,
        Module<B>: GLWEExternalProductInplace<B> + VmpPrepare<B>,
    {
        assert!(
            !self.subrams.is_empty(),
            "unitialized memory: self.data.len()=0"
        );

        self.subrams
            .iter_mut()
            .map(|subram| subram.read(&self.params, address, &keys.atk_glwe, self.scratch.borrow()))
            .collect()
    }

    /// Read that prepares the [Ram] of a subsequent [Self::write].
    /// Outside of preparing the [Ram] for a write, the Bhavior and
    /// output format is identical to [Self::read].
    pub fn read_prepare_write<D: DataRef, DA: DataRef>(
        &mut self,
        address: &Address<DA>,
        keys: &EvaluationKeysPrepared<D, B>,
    ) -> Vec<GLWECiphertext<Vec<u8>>>
    where
        ScratchOwned<B>: ScratchOwnedBorrow<B>,
        GGLWEAutomorphismKey<Vec<u8>>: PrepareAlloc<B, GGLWEAutomorphismKeyPrepared<Vec<u8>, B>>,
        Module<B>: VecZnxDftAllocBytes
            + VmpApplyDftToDftTmpBytes
            + VecZnxBigNormalizeTmpBytes
            + VmpApplyDftToDft<B>
            + VmpApplyDftToDftAdd<B>
            + VecZnxDftApply<B>
            + VecZnxIdftApplyConsume<B>
            + VecZnxBigAddSmallInplace<B>
            + VecZnxBigNormalize<B>
            + VecZnxCopy
            + VecZnxRotateInplace<B>
            + VecZnxSub
            + VecZnxNegateInplace
            + VecZnxRshInplace<B>
            + VecZnxAddInplace
            + VecZnxNormalizeInplace<B>
            + VecZnxSubInplace
            + VecZnxRotate
            + VecZnxAutomorphismInplace<B>
            + VecZnxBigSubSmallNegateInplace<B>
            + VecZnxBigAutomorphismInplace<B>
            + VecZnxNormalize<B>
            + VecZnxNormalizeTmpBytes
            + VmpPrepare<B>,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable + TakeVecZnx,
        Scratch<B>: TakeScalarZnx + TakeGGSWPrepared<B>,
        Module<B>: GLWEExternalProductInplace<B>,
    {
        assert!(
            !self.subrams.is_empty(),
            "unitialized memory: self.data.len()=0"
        );

        self.subrams
            .iter_mut()
            .map(|subram| {
                subram.read_prepare_write(
                    &self.params,
                    address,
                    &keys.atk_glwe,
                    self.scratch.borrow(),
                )
            })
            .collect()
    }

    /// Writes w to the [Ram]. Requires that [Self::read_prepare_write] was
    /// called Bforehand.
    pub fn write<D: DataRef, DA: DataRef, K: DataRef>(
        &mut self,
        w: &[GLWECiphertext<D>], // Must encrypt [w, 0, 0, ..., 0];
        address: &Address<DA>,
        keys: &EvaluationKeysPrepared<K, B>,
    ) where
        ScratchOwned<B>: ScratchOwnedBorrow<B>,
        Module<B>: VmpPMatAlloc<B> + VmpPrepare<B>,
        Scratch<B>: TakeScalarZnx,
        Module<B>: VecZnxDftAllocBytes
            + VmpApplyDftToDftTmpBytes
            + VecZnxBigNormalizeTmpBytes
            + VmpApplyDftToDft<B>
            + VmpApplyDftToDftAdd<B>
            + VecZnxDftApply<B>
            + VecZnxIdftApplyConsume<B>
            + VecZnxBigAddSmallInplace<B>
            + VecZnxBigNormalize<B>
            + VecZnxAutomorphismInplace<B>
            + VecZnxBigAllocBytes
            + VecZnxNormalizeTmpBytes
            + VecZnxDftCopy<B>
            + VecZnxDftAddInplace<B>
            + VecZnxIdftApplyTmpA<B>
            + VecZnxNormalize<B>,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable + TakeVecZnxBig<B> + TakeVecZnx,
        Module<B>: VecZnxNormalizeInplace<B> + VecZnxAddInplace + VecZnxSubInplace,
        Scratch<B>: TakeGLWECt + TakeGGSWPrepared<B>,
        Scratch<B>: TakeGGSWPrepared<B> + TakeGLWECt,
        Module<B>: VmpPrepare<B> + GLWEExternalProductInplace<B> + VecZnxSubInplace,
        Module<B>: VecZnxDftAllocBytes
            + VmpApplyDftToDftTmpBytes
            + VecZnxBigNormalizeTmpBytes
            + VmpApplyDftToDft<B>
            + VmpApplyDftToDftAdd<B>
            + VecZnxDftApply<B>
            + VecZnxIdftApplyConsume<B>
            + VecZnxBigAddSmallInplace<B>
            + VecZnxBigNormalize<B>
            + VecZnxBigAutomorphismInplace<B>
            + VecZnxRshInplace<B>
            + VecZnxCopy
            + VecZnxNormalizeTmpBytes
            + VecZnxNormalize<B>
            + VecZnxNormalizeInplace<B>
            + VecZnxAddInplace
            + VecZnxRotateInplace<B>
            + SvpApplyDftToDftInplace<B>
            + VecZnxBigAddInplace<B>,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable + TakeVecZnx,
        Scratch<B>: TakeGGSWPreparedSlice<B> + TakeGGSW,
    {
        assert!(w.len() == self.subrams.len());

        let params: &Parameters<B> = &self.params;
        let module: &Module<B> = params.module();

        let scratch: &mut Scratch<B> = self.scratch.borrow();
        let atk_glwe: &HashMap<i64, GGLWEAutomorphismKeyPrepared<K, B>> = &keys.atk_glwe;
        let atk_ggsw_inv: &GGLWEAutomorphismKeyPrepared<K, B> = &keys.atk_ggsw_inv;
        let tsk_ggsw_inv: &GGLWETensorKeyPrepared<K, B> = &keys.tsk_ggsw_inv;

        // Overwrites the coefficient that was read: to_write_on = to_write_on - TRACE(to_write_on) + w
        for (i, subram) in self.subrams.iter_mut().enumerate() {
            subram.write_first_step(params, &w[i], address.n2(), atk_glwe, scratch);
        }

        for i in (0..address.n2() - 1).rev() {
            // Index polynomial X^{i}
            let coordinate: &Coordinate<DA> = address.at(i + 1);

            let (mut inv_coordinate_prepared, scratch_1) =
                scratch.take_coordinate_prepared(coordinate, &coordinate.base1d);

            inv_coordinate_prepared.prepare_inv(
                module,
                coordinate,
                atk_ggsw_inv,
                tsk_ggsw_inv,
                scratch_1,
            );

            for subram in self.subrams.iter_mut() {
                subram.write_mid_step(i, params, &inv_coordinate_prepared, atk_glwe, scratch_1);
            }
        }

        let coordinate: &Coordinate<DA> = address.at(0);

        let (mut inv_coordinate_prepared, scratch_1) =
            scratch.take_coordinate_prepared(coordinate, &coordinate.base1d);

        inv_coordinate_prepared.prepare_inv(
            module,
            coordinate,
            atk_ggsw_inv,
            tsk_ggsw_inv,
            scratch_1,
        );

        for subram in self.subrams.iter_mut() {
            subram.write_last_step(module, &inv_coordinate_prepared, scratch_1);
        }
    }
}

/// [SubRam] stores a digit of the word.
pub struct SubRam {
    data: Vec<GLWECiphertext<Vec<u8>>>,
    tree: Vec<Vec<GLWECiphertext<Vec<u8>>>>,
    packer: GLWEPacker,
    state: bool,
}

impl SubRam {
    pub fn alloc<B: Backend>(params: &Parameters<B>) -> Self {
        let module: &Module<B> = params.module();

        let glwe_infos: GLWECiphertextLayout = params.glwe_ct_infos();

        let n: usize = module.n();
        let mut tree: Vec<Vec<GLWECiphertext<Vec<u8>>>> = Vec::new();
        let max_addr_split: usize = params.max_addr(); // u8 -> u32

        if max_addr_split > n {
            let mut size: usize = max_addr_split.div_ceil(n);
            while size != 1 {
                size = size.div_ceil(n);
                let tmp: Vec<GLWECiphertext<Vec<u8>>> = (0..size)
                    .map(|_| GLWECiphertext::alloc(&glwe_infos))
                    .collect();
                tree.push(tmp);
            }
        }

        Self {
            data: Vec::new(),
            tree,
            packer: GLWEPacker::new(&glwe_infos, 0),
            state: false,
        }
    }

    pub fn encrypt_sk<B: Backend>(
        &mut self,
        params: &Parameters<B>,
        data: &[u8],
        sk: &GLWESecret<Vec<u8>>,
        scratch: &mut Scratch<B>,
    ) where
        GLWESecret<Vec<u8>>: PrepareAlloc<B, GLWESecretPrepared<Vec<u8>, B>>,
        Module<B>: VecZnxDftAllocBytes
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
            + VecZnxSub,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable + TakeVecZnx,
    {
        let module: &Module<B> = params.module();

        let glwe_infos: GLWECiphertextLayout = params.glwe_ct_infos();
        let pt_infos: GLWECiphertextLayout = params.glwe_pt_infos();

        let mut source_xa: Source = Source::new([1u8; 32]); // TODO: Create from random seed
        let mut source_xe: Source = Source::new([1u8; 32]); // TODO: Create from random seed

        let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&pt_infos);
        let mut data_i64: Vec<i64> = vec![0i64; module.n()];
        let sk_glwe_prepared: GLWESecretPrepared<Vec<u8>, B> = sk.prepare_alloc(module, scratch);

        self.data = data
            .chunks(module.n())
            .map(|chunk| {
                let mut ct: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(&glwe_infos);
                izip!(data_i64.iter_mut(), chunk.iter())
                    .for_each(|(xi64, xu8)| *xi64 = *xu8 as i64);
                data_i64[chunk.len()..].iter_mut().for_each(|x| *x = 0);
                pt.encode_vec_i64(&data_i64, pt.k());
                ct.encrypt_sk(
                    module,
                    &pt,
                    &sk_glwe_prepared,
                    &mut source_xa,
                    &mut source_xe,
                    // sigma,
                    scratch,
                );
                ct
            })
            .collect();
    }

    fn read<K: DataRef, DA: DataRef, B: Backend>(
        &mut self,
        params: &Parameters<B>,
        address: &Address<DA>,
        auto_keys: &HashMap<i64, GGLWEAutomorphismKeyPrepared<K, B>>,
        scratch: &mut Scratch<B>,
    ) -> GLWECiphertext<Vec<u8>>
    where
        GGSWCiphertext<Vec<u8>>: PrepareAlloc<B, GGSWCiphertextPrepared<Vec<u8>, B>>,
        GGLWEAutomorphismKey<Vec<u8>>: PrepareAlloc<B, GGLWEAutomorphismKeyPrepared<Vec<u8>, B>>,
        Module<B>: VecZnxDftAllocBytes
            + VmpApplyDftToDftTmpBytes
            + VecZnxBigNormalizeTmpBytes
            + VmpApplyDftToDft<B>
            + VmpApplyDftToDftAdd<B>
            + VecZnxDftApply<B>
            + VecZnxIdftApplyConsume<B>
            + VecZnxBigAddSmallInplace<B>
            + VecZnxBigNormalize<B>
            + VecZnxCopy
            + VecZnxRotateInplace<B>
            + VecZnxSub
            + VecZnxNegateInplace
            + VecZnxRshInplace<B>
            + VecZnxAddInplace
            + VecZnxNormalizeInplace<B>
            + VecZnxSubInplace
            + VecZnxRotate
            + VecZnxAutomorphismInplace<B>
            + VecZnxBigSubSmallNegateInplace<B>
            + VecZnxBigAutomorphismInplace<B>
            + VecZnxNormalize<B>
            + VecZnxNormalizeTmpBytes,
        Scratch<B>: TakeVecZnxDft<B>
            + ScratchAvailable
            + TakeVecZnx
            + TakeScalarZnx
            + TakeCoordinatePrepared<B>,
        Module<B>: VmpPrepare<B>,
    {
        assert!(
            !self.state,
            "invalid call to Memory.read: internal state is true -> requires calling Memory.write"
        );

        let module: &Module<B> = params.module();
        let log_n: usize = module.log_n();

        let glwe_infos: GLWECiphertextLayout = params.glwe_ct_infos();
        let ggsw_infos: GGSWCiphertextLayout = params.ggsw_infos();

        assert_eq!(ggsw_infos, address.ggsw_layout());

        let packer: &mut GLWEPacker = &mut self.packer;

        let mut results: Vec<GLWECiphertext<Vec<u8>>> = Vec::new();
        let mut tmp_ct: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(&glwe_infos);

        for i in 0..address.n2() {
            let coordinate: &Coordinate<DA> = address.at(i);

            let res_prev: &Vec<GLWECiphertext<Vec<u8>>> =
                if i == 0 { &self.data } else { &results };

            let (mut coordinate_prepared, scratch_1) =
                scratch.take_coordinate_prepared(&ggsw_infos, &coordinate.base1d);

            coordinate_prepared.prepare(module, coordinate, scratch_1);

            if i < address.n2() - 1 {
                // let mut result_next: Vec<GLWECiphertext<Vec<u8>>> = Vec::new();

                for chunk in res_prev.chunks(module.n()) {
                    for j in 0..module.n() {
                        let j_rev = reverse_bits_msb(j, log_n as u32);

                        if j_rev < chunk.len() {
                            coordinate_prepared.product(
                                module,
                                &mut tmp_ct,
                                &chunk[j_rev],
                                scratch_1,
                            );
                            packer.add(module, Some(&tmp_ct), auto_keys, scratch_1);
                        } else {
                            packer.add(
                                module,
                                // &mut result_next,
                                None::<&GLWECiphertext<Vec<u8>>>,
                                auto_keys,
                                scratch_1,
                            );
                        }
                    }
                }

                packer.flush(module, &mut tmp_ct); //, auto_keys, scratch); // TODO: that to put instead of tmp_ct
                results.push(tmp_ct.clone());
            } else if i == 0 {
                coordinate_prepared.product(module, &mut tmp_ct, &self.data[0], scratch_1);
                results.push(tmp_ct.clone());
            } else {
                coordinate_prepared.product(module, &mut tmp_ct, &results[0], scratch_1);
            }
        }
        tmp_ct.trace_inplace(module, 0, log_n, auto_keys, scratch);
        tmp_ct
    }

    fn read_prepare_write<D: DataRef, DA: DataRef, B: Backend>(
        &mut self,
        params: &Parameters<B>,
        address: &Address<DA>,
        auto_keys: &HashMap<i64, GGLWEAutomorphismKeyPrepared<D, B>>,
        scratch: &mut Scratch<B>,
    ) -> GLWECiphertext<Vec<u8>>
    where
        Scratch<B>: TakeGGSWPrepared<B>,
        Module<B>: VmpPrepare<B> + GLWEExternalProductInplace<B>,
        Module<B>: VecZnxDftAllocBytes
            + VmpApplyDftToDftTmpBytes
            + VecZnxBigNormalizeTmpBytes
            + VmpApplyDftToDft<B>
            + VmpApplyDftToDftAdd<B>
            + VecZnxDftApply<B>
            + VecZnxIdftApplyConsume<B>
            + VecZnxBigAddSmallInplace<B>
            + VecZnxBigNormalize<B>
            + VecZnxCopy
            + VecZnxRotateInplace<B>
            + VecZnxSub
            + VecZnxNegateInplace
            + VecZnxRshInplace<B>
            + VecZnxAddInplace
            + VecZnxNormalizeInplace<B>
            + VecZnxSubInplace
            + VecZnxRotate
            + VecZnxAutomorphismInplace<B>
            + VecZnxBigSubSmallNegateInplace<B>
            + VecZnxBigAutomorphismInplace<B>
            + VecZnxNormalize<B>
            + VecZnxNormalizeTmpBytes,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable + TakeVecZnx,
    {
        assert!(
            !self.state,
            "invalid call to Memory.read: internal state is true -> requires calling Memory.write"
        );

        let module: &Module<B> = params.module();
        let log_n: usize = module.log_n();
        let ggsw_infos: GGSWCiphertextLayout = params.ggsw_infos();
        let packer: &mut GLWEPacker = &mut self.packer;

        assert_eq!(ggsw_infos, address.ggsw_layout());

        let mut results: Vec<GLWECiphertext<Vec<u8>>> = Vec::new();
        let mut tmp_ct: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(&params.glwe_ct_infos());

        for i in 0..address.n2() {
            let coordinate: &Coordinate<DA> = address.at(i);

            let res_prev: &mut Vec<GLWECiphertext<Vec<u8>>> = if i == 0 {
                &mut self.data
            } else {
                &mut self.tree[i - 1]
            };

            let (mut coordinate_prepared, scratch_1) =
                scratch.take_coordinate_prepared(&ggsw_infos, &coordinate.base1d);

            coordinate_prepared.prepare(module, coordinate, scratch_1);

            // Shift polynomial of the last iteration by X^{-i}
            for poly in res_prev.iter_mut() {
                coordinate_prepared.product_inplace(module, poly, scratch_1);
            }

            if i < address.n2() - 1 {
                // let mut result_next: Vec<GLWECiphertext<Vec<u8>>> = Vec::new();

                // Packs the first coefficient of each polynomial.
                for chunk in res_prev.chunks(module.n()) {
                    for i in 0..module.n() {
                        let i_rev: usize = reverse_bits_msb(i, log_n as u32);
                        if i_rev < chunk.len() {
                            packer.add(
                                module,
                                // &mut result_next,
                                Some(&chunk[i_rev]),
                                auto_keys,
                                scratch,
                            );
                        } else {
                            packer.add(
                                module,
                                // &mut result_next, // TODO : is it okay that this isn't Bing used?
                                None::<&GLWECiphertext<Vec<u8>>>,
                                auto_keys,
                                scratch,
                            );
                        }
                    }
                }

                packer.flush(module, &mut tmp_ct); //result_next, auto_keys, scratch);
                results.push(tmp_ct.clone());
                // packer.reset();

                // Stores the packed polynomial
                izip!(self.tree[i].iter_mut(), results.iter()).for_each(|(a, b)| {
                    a.copy(module, b);
                });
            }
        }

        let mut res: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(&params.glwe_ct_infos());

        self.state = true;
        if address.n2() != 1 {
            res.copy(module, &self.tree.last().unwrap()[0]);
        } else {
            res.copy(module, &self.data[0]);
        }

        res.trace_inplace(module, 0, log_n, auto_keys, scratch);
        res
    }

    fn write_first_step<DataW: DataRef, K: DataRef, B: Backend>(
        &mut self,
        params: &Parameters<B>,
        w: &GLWECiphertext<DataW>,
        n2: usize,
        auto_keys: &HashMap<i64, GGLWEAutomorphismKeyPrepared<K, B>>,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: VecZnxSubInplace + VecZnxAddInplace + VecZnxNormalizeInplace<B>,
        GGLWEAutomorphismKey<Vec<u8>>: PrepareAlloc<B, GGLWEAutomorphismKeyPrepared<Vec<u8>, B>>,
        Module<B>: VecZnxDftAllocBytes
            + VmpApplyDftToDftTmpBytes
            + VecZnxBigNormalizeTmpBytes
            + VmpApplyDftToDft<B>
            + VmpApplyDftToDftAdd<B>
            + VecZnxDftApply<B>
            + VecZnxIdftApplyConsume<B>
            + VecZnxBigAddSmallInplace<B>
            + VecZnxBigNormalize<B>
            + VecZnxBigAutomorphismInplace<B>
            + VecZnxRshInplace<B>
            + VecZnxCopy
            + VecZnxNormalizeTmpBytes
            + VecZnxNormalize<B>
            + SvpApplyDftToDftInplace<B>
            + VecZnxBigAddInplace<B>,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable + TakeVecZnx + TakeVecZnxBig<B>,
    {
        assert!(
            self.state,
            "invalid call to Memory.write: internal state is false -> requires calling Memory.read_prepare_write"
        );

        let module: &Module<B> = params.module();
        let log_n: usize = module.log_n();

        let glwe_infos: GLWECiphertextLayout = params.glwe_ct_infos();

        let to_write_on: &mut GLWECiphertext<Vec<u8>> = if n2 != 1 {
            &mut self.tree.last_mut().unwrap()[0]
        } else {
            &mut self.data[0]
        };

        let (mut tmp_a, scratch_1) = scratch.take_glwe_ct(&glwe_infos);
        tmp_a.trace(module, 0, log_n, to_write_on, auto_keys, scratch_1);

        to_write_on.sub_inplace_ab(module, &tmp_a);
        to_write_on.add_inplace(module, w);
        to_write_on.normalize_inplace(module, scratch_1);
    }

    fn write_mid_step<DC: DataRef, K: DataRef, B: Backend>(
        &mut self,
        step: usize,
        params: &Parameters<B>,
        inv_coordinate: &CoordinatePrepared<DC, B>,
        auto_keys: &HashMap<i64, GGLWEAutomorphismKeyPrepared<K, B>>,
        scratch: &mut Scratch<B>,
    ) where
        Scratch<B>: TakeGGSWPrepared<B> + TakeGLWECt,
        Module<B>: VmpPrepare<B> + GLWEExternalProductInplace<B> + VecZnxSubInplace,
        Module<B>: VecZnxDftAllocBytes
            + VmpApplyDftToDftTmpBytes
            + VecZnxBigNormalizeTmpBytes
            + VmpApplyDftToDft<B>
            + VmpApplyDftToDftAdd<B>
            + VecZnxDftApply<B>
            + VecZnxIdftApplyConsume<B>
            + VecZnxBigAddSmallInplace<B>
            + VecZnxBigNormalize<B>
            + VecZnxBigAutomorphismInplace<B>
            + VecZnxRshInplace<B>
            + VecZnxCopy
            + VecZnxNormalizeTmpBytes
            + VecZnxNormalize<B>
            + VecZnxNormalizeInplace<B>
            + VecZnxAddInplace
            + VecZnxRotateInplace<B>,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable + TakeVecZnx,
    {
        let module: &Module<B> = params.module();
        let log_n: usize = module.log_n();

        // Top of the tree is not stored in results.
        let (tree_hi, tree_lo) = if step == 0 {
            (&mut self.data, &mut self.tree[0])
        } else {
            let (left, right) = self.tree.split_at_mut(step);
            (&mut left[left.len() - 1], &mut right[0])
        };

        for (j, chunk) in tree_hi.chunks_mut(module.n()).enumerate() {
            // Retrieve the associated polynomial to extract and pack related to the current chunk
            let ct_lo: &mut GLWECiphertext<Vec<u8>> = &mut tree_lo[j];

            inv_coordinate.product_inplace(module, ct_lo, scratch);

            for ct_hi in chunk.iter_mut() {
                // Zeroes the first coefficient of ct_hi
                // ct_hi = [a, b, c, d] - TRACE([a, b, c, d]) = [0, b, c, d]
                let (mut tmp_a, scratch_1) = scratch.take_glwe_ct(&params.glwe_ct_infos());
                tmp_a.trace(module, 0, log_n, ct_hi, auto_keys, scratch_1);
                ct_hi.sub_inplace_ab(module, &tmp_a);

                // Extract the first coefficient ct_lo
                // tmp_a = TRACE([a, b, c, d]) -> [a, 0, 0, 0]
                tmp_a.trace(module, 0, log_n, ct_lo, auto_keys, scratch_1);

                // Adds extracted coefficient of ct_lo on ct_hi
                // [a, 0, 0, 0] + [0, b, c, d]
                ct_hi.add_inplace(module, &tmp_a);
                ct_hi.normalize_inplace(module, scratch_1);

                // Cyclic shift ct_lo by X^-1
                ct_lo.rotate_inplace(module, -1, scratch_1);
            }
        }
    }

    fn write_last_step<DC: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        inv_coordinate: &CoordinatePrepared<DC, B>,
        scratch: &mut Scratch<B>,
    ) where
        Scratch<B>: TakeGGSWPrepared<B>,
        Module<B>: VmpPrepare<B> + GLWEExternalProductInplace<B>,
    {
        // Apply the last reverse shift to the top of the tree.
        for ct_lo in self.data.iter_mut() {
            inv_coordinate.product_inplace(module, ct_lo, scratch);
        }

        self.state = false;
    }
}
