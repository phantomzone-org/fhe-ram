use std::time::Instant;

use poulpy_backend::FFT64Avx;
use poulpy_core::layouts::{
    GLWECiphertext, GLWECiphertextLayout, GLWEPlaintext, GLWESecret, LWEInfos,
    prepared::{GLWESecretPrepared, PrepareAlloc},
};
use poulpy_hal::{
    api::{
        ScratchAvailable, ScratchOwnedAlloc, ScratchOwnedBorrow, SvpApplyDftToDftInplace,
        TakeVecZnx, TakeVecZnxBig, TakeVecZnxDft, VecZnxAddInplace, VecZnxAddNormal,
        VecZnxBigAddInplace, VecZnxBigAddSmallInplace, VecZnxBigNormalize, VecZnxDftAllocBytes,
        VecZnxDftApply, VecZnxFillUniform, VecZnxIdftApplyConsume, VecZnxNormalize,
        VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes, VecZnxSub, VecZnxSubInplace,
    },
    layouts::{Backend, Module, Scratch, ScratchOwned},
    source::Source,
};

use fhe_ram::{Address, EvaluationKeys, Parameters, Ram};
use rand_core::RngCore;

fn main() {
    println!("Starting!");

    let seed_xs: [u8; 32] = [0u8; 32];
    let seed_xa: [u8; 32] = [0u8; 32];
    let seed_xe: [u8; 32] = [0u8; 32];

    let mut source_xs: Source = Source::new(seed_xs);
    let mut source_xa: Source = Source::new(seed_xa);
    let mut source_xe: Source = Source::new(seed_xe);

    // See parameters.rs for configuration
    let params: Parameters<FFT64Avx> = Parameters::<FFT64Avx>::new();

    // Generates a new secret-key along with the public evaluation keys.
    let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc(&params.glwe_ct_infos());
    sk.fill_ternary_prob(0.5, &mut source_xs);

    let keys: EvaluationKeys<Vec<u8>> =
        EvaluationKeys::encrypt_sk(&params, &sk, &mut source_xa, &mut source_xe);

    let mut scratch: ScratchOwned<_> = ScratchOwned::alloc(1 << 24);

    let sk_prep: GLWESecretPrepared<Vec<u8>, _> =
        sk.prepare_alloc(params.module(), scratch.borrow());

    let keys_prepared = keys.prepare_alloc(params.module(), scratch.borrow());

    // Some deterministic randomness
    let mut source: Source = Source::new([5u8; 32]);

    // Word-size
    let ws: usize = params.word_size();

    // Allocates some dummy data
    let mut data: Vec<u8> = vec![0u8; params.max_addr() * ws];
    source.fill_bytes(data.as_mut_slice());

    // Instantiates the FHE-RAM
    let mut ram: Ram<FFT64Avx> = Ram::new();

    // Populates the FHE-RAM
    ram.encrypt_sk(&data, &sk);

    // Allocates an encrypted address.
    let mut addr: Address<Vec<u8>> = Address::alloc(&params);

    // Random index
    let idx: u32 = source.next_u32() % params.max_addr() as u32;

    // Encrypts random index
    addr.encrypt_sk(&params, idx, &sk);

    // Reads from the FHE-RAM
    let start: Instant = Instant::now();
    let ct: Vec<GLWECiphertext<Vec<u8>>> = ram.read(&addr, &keys_prepared);
    let duration: std::time::Duration = start.elapsed();
    println!("READ Elapsed time: {} ms", duration.as_millis());

    // Checks correctness
    (0..ws).for_each(|i| {
        let want: i8 = data[i + ws * idx as usize] as i8;
        let (decrypted_value, noise) = decrypt_glwe(&params, &ct[i], want as u8, &sk_prep);
        assert_eq!(decrypted_value, want as i64);
        println!("noise: {}", noise);
        assert!(
            noise < -(params.k_glwe_pt().as_usize() as f64 + 1.0),
            "{} >= {}",
            noise,
            (params.k_glwe_pt().as_usize() as f64 + 1.0)
        );
    });

    // Reads from the FHE-RAM (with preparing for write)
    let start: Instant = Instant::now();
    let ct: Vec<GLWECiphertext<Vec<u8>>> = ram.read_prepare_write(&addr, &keys_prepared);
    let duration: std::time::Duration = start.elapsed();
    println!(
        "READ_PREPARE_WRITE Elapsed time: {} ms",
        duration.as_millis()
    );

    // Checks correctness
    (0..ws).for_each(|i| {
        let want: i8 = data[i + ws * idx as usize] as i8;
        let (decrypted_value, noise) = decrypt_glwe(&params, &ct[i], want as u8, &sk_prep);
        assert_eq!(decrypted_value, want as i64);
        println!("noise: {}", noise);
        assert!(
            noise < -(params.k_glwe_pt().as_usize() as f64 + 1.0),
            "{} >= {}",
            noise,
            (params.k_glwe_pt().as_usize() as f64 + 1.0)
        );
    });

    // Value to write on the FHE-RAM
    let mut value: Vec<u8> = vec![0u8; ws];
    source.fill_bytes(value.as_mut_slice());

    // Encryptes value to write on the FHE-RAM
    let ct_w = value
        .iter()
        .map(|wi| encrypt_glwe(&params, *wi, &sk_prep))
        .collect::<Vec<_>>();

    // Writes on the FHE-RAM
    let start: Instant = Instant::now();
    ram.write(&ct_w, &addr, &keys_prepared);
    let duration: std::time::Duration = start.elapsed();
    println!("WRITE Elapsed time: {} ms", duration.as_millis());

    // Updates plaintext ram
    (0..ws).for_each(|i| {
        data[i + ws * idx as usize] = value[i];
    });

    // Reads back at the written index
    let ct: Vec<GLWECiphertext<Vec<u8>>> = ram.read(&addr, &keys_prepared);

    // Checks correctness
    (0..ws).for_each(|i| {
        let want: i8 = data[i + ws * idx as usize] as i8;
        let (have, noise) = decrypt_glwe(&params, &ct[i], want as u8, &sk_prep);
        assert_eq!(have, want as i64);
        println!("noise: {}", noise);
        assert!(
            noise < -(params.k_glwe_pt().as_usize() as f64 + 1.0),
            "{} >= {}",
            noise,
            (params.k_glwe_pt().as_usize() as f64 + 1.0)
        );
    });
}

fn encrypt_glwe<B: Backend>(
    params: &Parameters<B>,
    value: u8,
    sk: &GLWESecretPrepared<Vec<u8>, B>,
) -> GLWECiphertext<Vec<u8>>
where
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
    Module<B>: VecZnxNormalizeTmpBytes + VecZnxDftAllocBytes,
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

    let mut ct_w: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(&glwe_infos);
    let mut pt_w: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&pt_infos);
    pt_w.encode_coeff_i64(value as i64, pt_infos.k(), 0);
    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(
        GLWECiphertext::encrypt_sk_scratch_space(module, &glwe_infos),
    );
    let mut source_xa: Source = Source::new([1u8; 32]); // TODO: Create from random seed
    let mut source_xe: Source = Source::new([1u8; 32]); // TODO: Create from random seed
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

fn decrypt_glwe<B: Backend>(
    params: &Parameters<B>,
    ct: &GLWECiphertext<Vec<u8>>,
    want: u8,
    sk: &GLWESecretPrepared<Vec<u8>, B>,
) -> (i64, f64)
where
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
    Module<B>: VecZnxDftAllocBytes + VecZnxNormalizeTmpBytes + VecZnxDftAllocBytes,
    GLWESecret<Vec<u8>>: PrepareAlloc<B, GLWESecretPrepared<Vec<u8>, B>>,
    Module<B>: VecZnxDftApply<B>
        + SvpApplyDftToDftInplace<B>
        + VecZnxIdftApplyConsume<B>
        + VecZnxBigAddInplace<B>
        + VecZnxBigAddSmallInplace<B>
        + VecZnxBigNormalize<B>,
    Scratch<B>: TakeVecZnxDft<B> + TakeVecZnxBig<B>,
{
    let module: &Module<B> = params.module();

    let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(ct);
    let mut scratch: ScratchOwned<B> =
        ScratchOwned::alloc(GLWECiphertext::decrypt_scratch_space(module, ct));

    ct.decrypt(module, &mut pt, sk, scratch.borrow());

    let log_scale: usize = pt.k().as_usize() - params.k_glwe_pt().as_usize();
    let decrypted_value: i64 = pt.decode_coeff_i64(pt.k(), 0);
    let diff: i64 = decrypted_value - (((want as i8) as i64) << log_scale);
    let noise: f64 = (diff.abs() as f64).log2() - pt.k().as_usize() as f64;
    (
        (decrypted_value as f64 / f64::exp2(log_scale as f64)).round() as i64,
        noise,
    )
}
