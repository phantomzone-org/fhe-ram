use itertools::Itertools;
use poulpy_backend::FFT64Spqlios;
use poulpy_core::layouts::{
    GLWECiphertext, GLWEPlaintext, Infos,
    prepared::{GLWESecretPrepared, PrepareAlloc},
};
use std::time::Instant;

use fhe_ram::{address::Address, keys::gen_keys, parameters::Parameters, ram::Ram};

use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Module, ScratchOwned},
    source::Source,
};
use rand_core::RngCore;

fn main() {
    // See parameters.rs for configuration
    let params: Parameters = Parameters::default();
    let module: &Module<FFT64Spqlios> = params.module();
    let basek: usize = params.basek();
    let k_ct: usize = params.k_ct();

    // Generates a new secret-key along with the public evaluation keys.
    let (sk_raw, keys) = gen_keys(&params);

    // Some randomness
    let mut source: Source = Source::new([0u8; 32]);

    // Scratch space.
    let mut scratch: ScratchOwned<FFT64Spqlios> = ScratchOwned::alloc(
        GLWECiphertext::encrypt_sk_scratch_space(module, basek, k_ct)
            | GLWECiphertext::decrypt_scratch_space(module, basek, k_ct),
    );

    // Prepare the secret.
    let sk_prep: GLWESecretPrepared<_, FFT64Spqlios> =
        sk_raw.prepare_alloc(module, scratch.borrow());

    // Word-size
    let ws: usize = params.word_size();

    // Allocates some dummy data
    let mut data: Vec<u8> = vec![0u8; params.max_addr() * ws];
    source.fill_bytes(data.as_mut_slice());

    // Instantiates the FHE-RAM
    let mut ram: Ram = Ram::default();

    // Populates the FHE-RAM
    ram.encrypt_sk(&data, &sk_raw);

    // Allocates an encrypted address.
    let mut addr: Address = Address::alloc(&params);

    // Random index
    let idx: u32 = source.next_u32() % params.max_addr() as u32;

    // Encrypts random index
    addr.encrypt_sk(&params, idx, &sk_prep);

    // Reads from the FHE-RAM
    let ct: Vec<GLWECiphertext<Vec<u8>>> = ram.read(&addr, &keys);

    // Checks correctness
    (0..ws).for_each(|i| {
        let want: u8 = data[i + idx as usize * ws];
        let noise: f64 = decrypt_glwe(&params, &ct[i], want, &sk_prep);
        println!("noise: {}", noise);
        assert!(
            noise < -(params.k_pt() as f64 + 1.0),
            "{} >= {}",
            noise,
            (params.k_pt() as f64 + 1.0)
        );
    });

    // Reads from the FHE-RAM (with preparing for write)
    let start: Instant = Instant::now();
    let ct: Vec<GLWECiphertext<Vec<u8>>> = ram.read_prepare_write(&addr, &keys);
    let duration: std::time::Duration = start.elapsed();
    println!("Elapsed time: {} ms", duration.as_millis());

    // Checks correctness
    (0..ws).for_each(|i| {
        let want: u8 = data[i + idx as usize * ws];
        let noise: f64 = decrypt_glwe(&params, &ct[i], want, &sk_prep);
        println!("noise: {}", noise);
        assert!(
            noise < -(params.k_pt() as f64 + 1.0),
            "{} >= {}",
            noise,
            (params.k_pt() as f64 + 1.0)
        );
    });

    // Value to write on the FHE-RAM
    let mut value: Vec<u8> = vec![0u8; ws];
    source.fill_bytes(value.as_mut_slice());

    // Encryptes value to write on the FHE-RAM
    let ct_w = value
        .iter()
        .map(|wi| encrypt_glwe(&params, *wi, &sk_prep))
        .collect_vec();

    // Writes on the FHE-RAM
    let start: Instant = Instant::now();
    ram.write(&ct_w, &addr, &keys);
    let duration: std::time::Duration = start.elapsed();
    println!("Elapsed time: {} ms", duration.as_millis());

    // Updates plaintext ram
    (0..ws).for_each(|i| {
        data[i + idx as usize * ws] = value[i];
    });

    // Reads back at the written index
    let ct: Vec<GLWECiphertext<Vec<u8>>> = ram.read(&addr, &keys);

    // Checks correctness
    (0..ws).for_each(|i| {
        let want: u8 = data[i + idx as usize * ws];
        let noise: f64 = decrypt_glwe(&params, &ct[i], want, &sk_prep);
        println!("noise: {}", noise);
        assert!(
            noise < -(params.k_pt() as f64 + 1.0),
            "{} >= {}",
            noise,
            (params.k_pt() as f64 + 1.0)
        );
    });
}

fn encrypt_glwe(
    params: &Parameters,
    value: u8,
    sk: &GLWESecretPrepared<Vec<u8>, FFT64Spqlios>,
) -> GLWECiphertext<Vec<u8>> {
    let module: &Module<FFT64Spqlios> = params.module();
    let basek: usize = params.basek();
    let k_ct: usize = params.k_ct();
    let k_pt: usize = params.k_pt();
    let rank: usize = params.rank();
    let mut ct_w: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(module.n(), basek, k_ct, rank);
    let mut pt_w: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(module.n(), basek, k_pt);
    pt_w.data
        .encode_coeff_i64(basek, 0, k_pt, 0, value as i64, u8::BITS as usize);
    let mut scratch: ScratchOwned<FFT64Spqlios> = ScratchOwned::alloc(
        GLWECiphertext::encrypt_sk_scratch_space(module, basek, ct_w.k()),
    );
    let mut source_xa: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([1u8; 32]);
    ct_w.encrypt_sk(
        module,
        &pt_w,
        sk,
        &mut source_xa,
        &mut source_xe,
        scratch.borrow(),
    );
    ct_w
}

fn decrypt_glwe(
    params: &Parameters,
    ct: &GLWECiphertext<Vec<u8>>,
    want: u8,
    sk: &GLWESecretPrepared<Vec<u8>, FFT64Spqlios>,
) -> f64 {
    let module: &Module<FFT64Spqlios> = params.module();
    let basek: usize = params.basek();
    let k: usize = ct.k();
    let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(module.n(), basek, k);
    let mut scratch: ScratchOwned<FFT64Spqlios> =
        ScratchOwned::alloc(GLWECiphertext::decrypt_scratch_space(module, basek, ct.k()));
    ct.decrypt(module, &mut pt, sk, scratch.borrow());
    let mut value: i64 = pt.data.decode_coeff_i64(basek, 0, k, 0);
    value -= ((want as i8) as i64) << (k - params.k_pt());
    let noise: f64 = ((value).abs() as f64).log2() - (k as f64);
    noise
}
