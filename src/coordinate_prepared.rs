use poulpy_hal::{
    api::{
        ScratchAvailable, TakeScalarZnx, TakeVecZnx, TakeVecZnxBig, TakeVecZnxDft,
        VecZnxAutomorphismInplace, VecZnxBigAddSmallInplace, VecZnxBigAllocBytes,
        VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxDftAddInplace, VecZnxDftAllocBytes,
        VecZnxDftApply, VecZnxDftCopy, VecZnxIdftApplyConsume, VecZnxIdftApplyTmpA,
        VecZnxNormalize, VecZnxNormalizeTmpBytes, VmpApplyDftToDft, VmpApplyDftToDftAdd,
        VmpApplyDftToDftTmpBytes, VmpPMatAllocBytes, VmpPrepare,
    },
    layouts::{Backend, Data, DataMut, DataRef, Module, Scratch},
};

use poulpy_core::{
    GLWEExternalProduct, GLWEExternalProductInplace, TakeGGSW, TakeGGSWPreparedSlice,
    layouts::{
        GGSWCiphertext, GGSWInfos, GLWECiphertext, GLWEInfos, LWEInfos,
        prepared::{
            GGLWEAutomorphismKeyPrepared, GGLWETensorKeyPrepared, GGSWCiphertextPrepared, Prepare,
        },
    },
};

use crate::{Base1D, Coordinate};

pub(crate) struct CoordinatePrepared<D: Data, B: Backend> {
    pub(crate) value: Vec<GGSWCiphertextPrepared<D, B>>,
    pub(crate) base1d: Base1D,
}

impl<B: Backend> CoordinatePrepared<Vec<u8>, B>
where
    Module<B>: VmpPMatAllocBytes,
{
    pub(crate) fn alloc_bytes<A>(module: &Module<B>, infos: &A, size: usize) -> usize
    where
        A: GGSWInfos,
    {
        size * GGSWCiphertextPrepared::alloc_bytes(module, infos)
    }
}

impl<D: Data, B: Backend> LWEInfos for CoordinatePrepared<D, B> {
    fn base2k(&self) -> poulpy_core::layouts::Base2K {
        self.value[0].base2k()
    }

    fn k(&self) -> poulpy_core::layouts::TorusPrecision {
        self.value[0].k()
    }

    fn n(&self) -> poulpy_core::layouts::Degree {
        self.value[0].n()
    }
}

impl<D: Data, B: Backend> GLWEInfos for CoordinatePrepared<D, B> {
    fn rank(&self) -> poulpy_core::layouts::Rank {
        self.value[0].rank()
    }
}

impl<D: Data, B: Backend> GGSWInfos for CoordinatePrepared<D, B> {
    fn dnum(&self) -> poulpy_core::layouts::Dnum {
        self.value[0].dnum()
    }

    fn dsize(&self) -> poulpy_core::layouts::Dsize {
        self.value[0].dsize()
    }
}

pub(crate) trait TakeCoordinatePrepared<B: Backend> {
    fn take_coordinate_prepared<A>(
        &mut self,
        infos: &A,
        base1d: &Base1D,
    ) -> (CoordinatePrepared<&mut [u8], B>, &mut Self)
    where
        A: GGSWInfos;
}

impl<B: Backend> TakeCoordinatePrepared<B> for Scratch<B>
where
    Scratch<B>: TakeGGSWPreparedSlice<B>,
{
    fn take_coordinate_prepared<A>(
        &mut self,
        infos: &A,
        base1d: &Base1D,
    ) -> (CoordinatePrepared<&mut [u8], B>, &mut Self)
    where
        A: GGSWInfos,
    {
        let (ggsws, scratch) = self.take_ggsw_prepared_slice(base1d.0.len(), infos);
        (
            CoordinatePrepared {
                value: ggsws,
                base1d: base1d.clone(),
            },
            scratch,
        )
    }
}

impl<DM: DataMut, DR: DataRef, B: Backend> Prepare<B, Coordinate<DR>> for CoordinatePrepared<DM, B>
where
    GGSWCiphertextPrepared<DM, B>: Prepare<B, GGSWCiphertext<DR>>,
{
    fn prepare(&mut self, module: &Module<B>, other: &Coordinate<DR>, scratch: &mut Scratch<B>) {
        assert_eq!(self.base1d, other.base1d);
        for (el_prep, el) in self.value.iter_mut().zip(other.value.iter()) {
            el_prep.prepare(module, el, scratch)
        }
    }
}

impl<D: DataMut, B: Backend> CoordinatePrepared<D, B> {
    /// Maps GGSW(X^{i}) to GGSW(X^{-i}).
    pub(crate) fn prepare_inv<DataOther: DataRef, DataAK: DataRef, DataTK: DataRef>(
        &mut self,
        module: &Module<B>,
        other: &Coordinate<DataOther>,
        auto_key: &GGLWEAutomorphismKeyPrepared<DataAK, B>,
        tensor_key: &GGLWETensorKeyPrepared<DataTK, B>,
        scratch: &mut Scratch<B>,
    ) where
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
            + VecZnxNormalize<B>
            + VmpPrepare<B>,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable + TakeVecZnxBig<B> + TakeVecZnx + TakeGGSW,
    {
        assert!(auto_key.p() == -1);
        assert_eq!(self.base1d, other.base1d);
        let (mut tmp_ggsw, scratch_1) = scratch.take_ggsw(other);
        for (prepared, other) in self.value.iter_mut().zip(other.value.iter()) {
            tmp_ggsw.automorphism(module, other, auto_key, tensor_key, scratch_1);
            prepared.prepare(module, &tmp_ggsw, scratch_1);
        }
        self.base1d = other.base1d.clone();
    }
}

impl<D: DataRef, B: Backend> CoordinatePrepared<D, B> {
    /// Evaluates GLWE(m) * GGSW(X^i).
    pub(crate) fn product<DataRes: DataMut, DataA: DataRef>(
        &self,
        module: &Module<B>,
        res: &mut GLWECiphertext<DataRes>,
        a: &GLWECiphertext<DataA>,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: GLWEExternalProduct<B> + GLWEExternalProductInplace<B>,
    {
        for (i, coordinate) in self.value.iter().enumerate() {
            if i == 0 {
                res.external_product(module, a, coordinate, scratch);
            } else {
                res.external_product_inplace(module, coordinate, scratch);
            }
        }
    }

    /// Evaluates GLWE(m) * GGSW(X^i).
    pub(crate) fn product_inplace<RES: DataMut>(
        &self,
        module: &Module<B>,
        res: &mut GLWECiphertext<RES>,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: GLWEExternalProductInplace<B>,
    {
        for coordinate in self.value.iter() {
            res.external_product_inplace(module, coordinate, scratch);
        }
    }
}
