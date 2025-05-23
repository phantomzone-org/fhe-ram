use core::{
    automorphism::AutomorphismKey,
    glwe_ciphertext::GLWECiphertext,
    keys::{SecretKey, SecretKeyFourier},
    tensor_key::TensorKey,
};
use std::collections::HashMap;

use backend::{Module, ScalarZnx, ScalarZnxToRef, ScratchOwned, FFT64};
use sampling::source::{Source, new_seed};

use crate::{address::{get_base_2d, Address, Decomp}, parameters::Parameters};

pub struct EvaluationKeys {
    pub(crate) auto_keys: HashMap<i64, AutomorphismKey<Vec<u8>, FFT64>>,
    pub(crate) tensor_key: TensorKey<Vec<u8>, FFT64>,
}

pub struct Client {
    params: Parameters,
    max_addr: u32,
    decomp: Decomp,
    sk: SecretKey<Vec<u8>>,
}

impl Client {
    pub fn new(max_addr: u32) -> (Client, EvaluationKeys) {

        let params: Parameters = Parameters::new();

        let d2 = get_base_2d(max_addr, params.decomp_n);

        let module: &Module<FFT64> = &params.module;
        let basek: usize = params.basek;
        let k_evk: usize = params.k_evk;
        let size_evk: usize = params.size_evk;
        let rows: usize = params.rows;
        let rank: usize = params.rank;

        let mut source_1: Source = Source::new(new_seed());
        let mut source_2: Source = Source::new(new_seed());

        let mut scratch: ScratchOwned = ScratchOwned::new(
            AutomorphismKey::generate_from_sk_scratch_space(module, rank, size_evk)
                | TensorKey::generate_from_sk_scratch_space(module, rank, size_evk),
        );

        let mut sk: SecretKey<Vec<u8>> = SecretKey::alloc(module, params.rank);
        sk.fill_ternary_prob(params.xs, &mut source_1);

        let mut sk_dft: SecretKeyFourier<Vec<u8>, FFT64> = SecretKeyFourier::alloc(module, rank);
        sk_dft.dft(module, &sk);

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
                params.xe,
                scratch.borrow(),
            );
            auto_keys.insert(*gal_el, key);
        });

        let mut tensor_key = TensorKey::alloc(module, basek, k_evk, rows, rank);
        tensor_key.generate_from_sk(
            module,
            &sk_dft,
            &mut source_1,
            &mut source_2,
            params.xe,
            scratch.borrow(),
        );

        (Client {
            params: params,
            decomp,
            max_addr,
            sk: sk,
        }, EvaluationKeys {
            auto_keys,
            tensor_key,
        })
    }

    pub fn gen_address<D>(&self, sk: &SecretKey<D>, value: u32) -> Address where ScalarZnx<D>: ScalarZnxToRef{
        let params: &Parameters = &self.params;
        let module: &Module<FFT64> = &self.params.module;
        let basek: usize = params.basek;
        let k_evk: usize = params.k_evk;
        let size_evk: usize = params.size_evk;
        let rows: usize = params.rows;
        let rank: usize = params.rank;

        

        let decomp = max_addr

        let mut adr: Address = Address::new(module, decomp, basek, k_evk, rows, rank);


        adr
    }


}
