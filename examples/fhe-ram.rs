use core::{
    elem::Infos,
    glwe_ciphertext::GLWECiphertext,
    glwe_plaintext::GLWEPlaintext,
    keys::{SecretKey, SecretKeyFourier},
};

use backend::{Decoding, Encoding, FFT64, Module, ScratchOwned};
use fhe_ram::{address::Address, keys::gen_keys, parameters::Parameters, ram::Ram};
use rand_core::RngCore;
use sampling::source::{Source, new_seed};

fn main() {
    let params: Parameters = Parameters::new();
    let (sk, keys) = gen_keys(&params);

    let mut source: Source = Source::new([0u8; 32]);

    let mut data: Vec<u8> = vec![0u8; params.max_addr()];
    data.iter_mut().for_each(|x| *x = source.next_u32() as u8);

    let mut ram: Ram = Ram::new();

    ram.encrypt_sk(&data, &sk);

    let mut addr: Address = Address::alloc(&params);

    let idx: u32 = 127;

    addr.encrypt_sk(&params, idx, &sk);

    let ct: GLWECiphertext<Vec<u8>> = ram.read(&addr, &keys);
    let want: u8 = data[idx as usize];
    let noise: f64 = decrypt_glwe(&params, &ct, want, &sk);
    println!("noise: {}", noise);
    assert!(
        noise < -(params.k_pt() as f64 + 1.0),
        "{} >= {}",
        noise,
        (params.k_pt() as f64 + 1.0)
    );

    let ct: GLWECiphertext<Vec<u8>> = ram.read_prepare_write(&addr, &keys);
    let want: u8 = data[idx as usize];
    let noise: f64 = decrypt_glwe(&params, &ct, want, &sk);
    println!("noise: {}", noise);
    assert!(
        noise < -(params.k_pt() as f64 + 1.0),
        "{} >= {}",
        noise,
        (params.k_pt() as f64 + 1.0)
    );
    let value: u8 = 4;

    let ct_w: GLWECiphertext<Vec<u8>> = encrypt_glwe(&params, value, &sk);

    ram.write(&ct_w, &addr, &keys, &sk);
    data[idx as usize] = value;

    let ct: GLWECiphertext<Vec<u8>> = ram.read(&addr, &keys);
    let want: u8 = data[idx as usize];
    let noise: f64 = decrypt_glwe(&params, &ct, want, &sk);
    println!("noise: {}", noise);
    assert!(
        noise < -(params.k_pt() as f64 + 1.0),
        "{} >= {}",
        noise,
        (params.k_pt() as f64 + 1.0)
    );
}

fn encrypt_glwe(
    params: &Parameters,
    value: u8,
    sk: &SecretKey<Vec<u8>>,
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
        ct_w.size(),
    ));
    let mut source_xa: Source = Source::new(new_seed());
    let mut source_xe: Source = Source::new(new_seed());
    let mut sk_dft: SecretKeyFourier<Vec<u8>, FFT64> = SecretKeyFourier::alloc(module, rank);
    sk_dft.dft(module, &sk);
    ct_w.encrypt_sk(
        module,
        &pt_w,
        &sk_dft,
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
    sk: &SecretKey<Vec<u8>>,
) -> f64 {
    let module: &Module<FFT64> = params.module();
    let basek: usize = params.basek();
    let k: usize = ct.k();
    let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(module, basek, k);
    let mut scratch: ScratchOwned =
        ScratchOwned::new(GLWECiphertext::decrypt_scratch_space(module, ct.size()));
    let mut sk_dft: SecretKeyFourier<Vec<u8>, backend::FFT64> =
        SecretKeyFourier::alloc(module, sk.rank());
    sk_dft.dft(module, &sk);
    ct.decrypt(module, &mut pt, &sk_dft, scratch.borrow());
    let mut value: i64 = pt.data.decode_coeff_i64(0, basek, k, 0);
    value -= (want as i64) << (k - params.k_pt());
    let noise: f64 = ((value).abs() as f64).log2() - (k as f64);
    noise
}
