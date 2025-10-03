use std::time::Instant;

use poulpy_core::layouts::{
    GLWECiphertext, GLWEPlaintext, GLWESecret, Infos,
    prepared::{GLWESecretPrepared, PrepareAlloc},
};
use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Module, ScratchOwned},
    source::Source,
};

use fhe_ram::{BackendImpl, address::Address, keys::gen_keys, parameters::Parameters, ram::Ram};
use rand_core::RngCore;

fn main() {
    println!("Starting!");

    // See parameters.rs for configuration
    let params: Parameters = Parameters::new();

    // Generates a new secret-key along with the public evaluation keys.
    let (sk, keys) = gen_keys(&params);

    // Some deterministic randomness
    let mut source: Source = Source::new([5u8; 32]);

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
    let start: Instant = Instant::now();
    let ct: Vec<GLWECiphertext<Vec<u8>>> = ram.read(&addr, &keys);
    let duration: std::time::Duration = start.elapsed();
    println!("READ Elapsed time: {} ms", duration.as_millis());

    // Checks correctness
    (0..ws).for_each(|i| {
        let want: i8 = data[i + ws * idx as usize] as i8;
        let (decrypted_value, noise) = decrypt_glwe(&params, &ct[i], want as u8, &sk);
        assert_eq!(decrypted_value, want as i64);
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
    println!(
        "READ_PREPARE_WRITE Elapsed time: {} ms",
        duration.as_millis()
    );

    // Checks correctness
    (0..ws).for_each(|i| {
        let want: i8 = data[i + ws * idx as usize] as i8;
        let (decrypted_value, noise) = decrypt_glwe(&params, &ct[i], want as u8, &sk);
        assert_eq!(decrypted_value, want as i64);
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
        .collect::<Vec<_>>();

    // Writes on the FHE-RAM
    let start: Instant = Instant::now();
    ram.write(&ct_w, &addr, &keys);
    let duration: std::time::Duration = start.elapsed();
    println!("WRITE Elapsed time: {} ms", duration.as_millis());

    // Updates plaintext ram
    (0..ws).for_each(|i| {
        data[i + ws * idx as usize] = value[i];
    });

    // Reads back at the written index
    let ct: Vec<GLWECiphertext<Vec<u8>>> = ram.read(&addr, &keys);

    // Checks correctness
    (0..ws).for_each(|i| {
        let want: i8 = data[i + ws * idx as usize] as i8;
        let (have, noise) = decrypt_glwe(&params, &ct[i], want as u8, &sk);
        assert_eq!(have, want as i64);
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
    sk: &GLWESecret<Vec<u8>>,
) -> GLWECiphertext<Vec<u8>> {
    let module: &Module<BackendImpl> = params.module();
    let basek: usize = params.basek();
    let k_ct: usize = params.k_ct();
    let k_pt: usize = params.k_pt();
    let rank: usize = params.rank();
    let mut ct_w: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(module.n(), basek, k_ct, rank);
    let mut pt_w: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(module.n(), basek, k_pt);
    pt_w.data
        .encode_coeff_i64(basek, 0, k_pt, 0, value as i64, u8::BITS as usize);
    let mut scratch: ScratchOwned<BackendImpl> = ScratchOwned::alloc(
        GLWECiphertext::encrypt_sk_scratch_space(module, basek, ct_w.k()),
    );
    let mut source_xa: Source = Source::new([1u8; 32]); // TODO: Create from random seed
    let mut source_xe: Source = Source::new([1u8; 32]); // TODO: Create from random seed
    let sk_glwe_prepared: GLWESecretPrepared<Vec<u8>, BackendImpl> =
        sk.prepare_alloc(&module, scratch.borrow());
    ct_w.encrypt_sk(
        module,
        &pt_w,
        &sk_glwe_prepared,
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
    sk: &GLWESecret<Vec<u8>>,
) -> (i64, f64) {
    let module: &Module<BackendImpl> = params.module();
    let basek: usize = params.basek();
    let k: usize = ct.k();
    let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(module.n(), basek, k);
    let mut scratch: ScratchOwned<BackendImpl> =
        ScratchOwned::alloc(GLWECiphertext::decrypt_scratch_space(module, basek, ct.k()));

    let sk_glwe_prepared: GLWESecretPrepared<Vec<u8>, BackendImpl> =
        sk.prepare_alloc(&module, scratch.borrow());
    ct.decrypt(module, &mut pt, &sk_glwe_prepared, scratch.borrow());

    let log_scale: usize = k - params.k_pt();
    let decrypted_value: i64 = pt.data.decode_coeff_i64(basek, 0, k, 0);
    let diff: i64 = decrypted_value - (((want as i8) as i64) << log_scale);
    let noise: f64 = (diff.abs() as f64).log2() - k as f64;
    (
        (decrypted_value as f64 / f64::exp2(log_scale as f64)).round() as i64,
        noise,
    )
}
