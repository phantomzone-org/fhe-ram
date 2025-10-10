use itertools::izip;
use poulpy_hal::{
    api::{
        ScratchAvailable, ScratchOwnedAlloc, ScratchOwnedBorrow, SvpApplyDftToDftInplace,
        TakeScalarZnx, TakeVecZnx, TakeVecZnxDft, VecZnxAddInplace, VecZnxAddNormal,
        VecZnxAddScalarInplace, VecZnxBigNormalize, VecZnxDftAllocBytes, VecZnxDftApply,
        VecZnxFillUniform, VecZnxIdftApplyConsume, VecZnxNormalize, VecZnxNormalizeInplace,
        VecZnxNormalizeTmpBytes, VecZnxSub, VecZnxSubInplace,
    },
    layouts::{Backend, Data, DataMut, Module, Scratch, ScratchOwned},
    source::Source,
};

use poulpy_core::layouts::{
    GGSWCiphertext, GGSWCiphertextLayout, GGSWInfos, GLWEInfos, GLWESecret, LWEInfos,
    prepared::{GLWESecretPrepared, PrepareAlloc},
};
use rand_core::{OsRng, TryRngCore};

use crate::{Base2D, Coordinate, parameters::Parameters};

/// [Address] stores GGSW(X^{addr}) in decomposed
/// form. That is, given addr = prod X^{a_i}, then
/// it stores Vec<[Coordinate]:(X^{a_0}), [Coordinate]:(X^{a_1}), ...>.
/// where [a_0, a_1, ...] is the representation in base N of a.
///
/// Such decomposition is necessary if the ring degree
/// N is smaller than the maximum supported address.
pub struct Address<D: Data> {
    pub coordinates: Vec<Coordinate<D>>,
    pub base2d: Base2D,
}

impl<D: Data> LWEInfos for Address<D> {
    fn base2k(&self) -> poulpy_core::layouts::Base2K {
        self.coordinates[0].base2k()
    }

    fn k(&self) -> poulpy_core::layouts::TorusPrecision {
        self.coordinates[0].k()
    }

    fn n(&self) -> poulpy_core::layouts::Degree {
        self.coordinates[0].n()
    }
}

impl<D: Data> GLWEInfos for Address<D> {
    fn rank(&self) -> poulpy_core::layouts::Rank {
        self.coordinates[0].rank()
    }
}

impl<D: Data> GGSWInfos for Address<D> {
    fn dnum(&self) -> poulpy_core::layouts::Dnum {
        self.coordinates[0].dnum()
    }

    fn dsize(&self) -> poulpy_core::layouts::Dsize {
        self.coordinates[0].dsize()
    }
}

impl Address<Vec<u8>> {
    /// Allocates a new [Address].
    pub fn alloc<B: Backend>(params: &Parameters<B>) -> Self {
        let base_2d: Base2D = params.base2d();
        let ggsw_infos: GGSWCiphertextLayout = params.ggsw_infos();

        Self {
            coordinates: base_2d
                .0
                .iter()
                .map(|base1d| Coordinate::alloc(&ggsw_infos, base1d))
                .collect(),
            base2d: base_2d.clone(),
        }
    }
}

impl<D: DataMut> Address<D> {
    /// Encrypts an u32 value into an [Address] under the provided secret.
    pub fn encrypt_sk<B: Backend>(
        &mut self,
        params: &Parameters<B>,
        value: u32,
        sk: &GLWESecret<Vec<u8>>,
    ) where
        Module<B>: VecZnxNormalizeTmpBytes + VecZnxDftAllocBytes,
        ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
        Scratch<B>: TakeScalarZnx,
        GLWESecret<Vec<u8>>: PrepareAlloc<B, GLWESecretPrepared<Vec<u8>, B>>,
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
            + VecZnxSub,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable + TakeVecZnx,
    {
        debug_assert!(self.base2d.max() > value as usize);

        let module: &Module<B> = params.module();

        let ggsw_infos: GGSWCiphertextLayout = params.ggsw_infos();

        let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(
            GGSWCiphertext::encrypt_sk_scratch_space(module, &ggsw_infos),
        );

        let mut seed_xa: [u8; 32] = [0u8; 32];
        OsRng.try_fill_bytes(&mut seed_xa).unwrap();

        let mut seed_xe: [u8; 32] = [0u8; 32];
        OsRng.try_fill_bytes(&mut seed_xe).unwrap();

        let mut source_xa: Source = Source::new(seed_xa);
        let mut source_xe: Source = Source::new(seed_xe);

        let mut remain: usize = value as _;
        izip!(self.coordinates.iter_mut(), self.base2d.0.iter()).for_each(|(coordinate, base1d)| {
            let max: usize = base1d.max();
            let k: usize = remain & (max - 1);
            coordinate.encrypt_sk(
                -(k as i64),
                module,
                sk,
                &mut source_xa,
                &mut source_xe,
                scratch.borrow(),
            );
            remain /= max;
        })
    }
}

impl<D: Data> Address<D> {
    pub(crate) fn n2(&self) -> usize {
        self.coordinates.len()
    }

    pub(crate) fn at(&self, i: usize) -> &Coordinate<D> {
        &self.coordinates[i]
    }
}
