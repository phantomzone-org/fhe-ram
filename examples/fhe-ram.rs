use core::{GLWECiphertext, GLWEPlaintext, GLWESecret, Infos};
use std::time::Instant;

use backend::{Decoding, Encoding, FFT64, Module, ScratchOwned};
use fhe_ram::{address::Address, keys::gen_keys, parameters::Parameters, ram::Ram};
use itertools::Itertools;
use rand_core::RngCore;
use sampling::source::{Source, new_seed};

fn main() {
    // See parameters.rs for configuration
    let params: Parameters = Parameters::new();

    // Generates a new secret-key along with the public evaluation keys.
    let (sk, keys) = gen_keys(&params);

    // Some randomness
    let mut source: Source = Source::new([0u8; 32]);

    // Word-size
    let ws: usize = params.word_size();

    // Allocates some dummy data
    let mut data: Vec<u8> = vec![0u8; params.max_addr() * ws];
    source.fill_bytes(data.as_mut_slice());

    // Instantiates the FHE-RAM
    let mut ram: Ram = Ram::new();

    // Populates the FHE-RAM
    ram.encrypt_sk(&data, &sk);

    // Allocates an encrypted address.
    let mut addr: Address = Address::alloc(&params);

    // Random index
    let idx: u32 = source.next_u32() % params.max_addr() as u32;

    // Encrypts random index
    addr.encrypt_sk(&params, idx, &sk);

    // Reads from the FHE-RAM
    let ct: Vec<GLWECiphertext<Vec<u8>>> = ram.read(&addr, &keys);

    // Checks correctness
    (0..ws).for_each(|i| {
        let want: u8 = data[i + idx as usize];
        let noise: f64 = decrypt_glwe(&params, &ct[i], want, &sk);
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
        let want: u8 = data[i + idx as usize];
        let noise: f64 = decrypt_glwe(&params, &ct[i], want, &sk);
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
        .map(|wi| encrypt_glwe(&params, *wi, &sk))
        .collect_vec();

    // Writes on the FHE-RAM
    let start: Instant = Instant::now();
    ram.write(&ct_w, &addr, &keys);
    let duration: std::time::Duration = start.elapsed();
    println!("Elapsed time: {} ms", duration.as_millis());

    // Updates plaintext ram
    (0..ws).for_each(|i| {
        data[i + idx as usize] = value[i];
    });

    // Reads back at the written index
    let ct: Vec<GLWECiphertext<Vec<u8>>> = ram.read(&addr, &keys);

    // Checks correctness
    (0..ws).for_each(|i| {
        let want: u8 = data[i + idx as usize];
        let noise: f64 = decrypt_glwe(&params, &ct[i], want, &sk);
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
    sk: &GLWESecret<Vec<u8>, FFT64>,
) -> GLWECiphertext<Vec<u8>> {
    let module: &Module<FFT64> = params.module();
    let basek: usize = params.basek();
    let k_ct: usize = params.k_ct();
    let k_pt: usize = params.k_pt();
    let rank: usize = params.rank();
    let mut ct_w: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(module, basek, k_ct, rank);
    let mut pt_w: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(module, basek, k_pt);
    pt_w.data
        .encode_coeff_i64(0, basek, k_pt, 0, value as i64, u8::BITS as usize);
    let mut scratch: ScratchOwned = ScratchOwned::new(GLWECiphertext::encrypt_sk_scratch_space(
        module,
        basek,
        ct_w.k(),
    ));
    let mut source_xa: Source = Source::new(new_seed());
    let mut source_xe: Source = Source::new(new_seed());
    ct_w.encrypt_sk(
        module,
        &pt_w,
        &sk,
        &mut source_xa,
        &mut source_xe,
        params.xe(),
        scratch.borrow(),
    );
    ct_w
}

fn decrypt_glwe(
    params: &Parameters,
    ct: &GLWECiphertext<Vec<u8>>,
    want: u8,
    sk: &GLWESecret<Vec<u8>, FFT64>,
) -> f64 {
    let module: &Module<FFT64> = params.module();
    let basek: usize = params.basek();
    let k: usize = ct.k();
    let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(module, basek, k);
    let mut scratch: ScratchOwned =
        ScratchOwned::new(GLWECiphertext::decrypt_scratch_space(module, basek, ct.k()));
    ct.decrypt(module, &mut pt, &sk, scratch.borrow());
    let mut value: i64 = pt.data.decode_coeff_i64(0, basek, k, 0);
    value -= ((want as i8) as i64) << (k - params.k_pt());
    let noise: f64 = ((value).abs() as f64).log2() - (k as f64);
    noise
}
