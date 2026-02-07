# Discontinued

This crate has been discontinued in favor of a better implementation in the [Phantom VM](https://github.com/phantomzone-org/phantom/blob/main/fhevm/src/memory.rs).  
It is no longer maintained and will not compile with recent Poulpy versions.

If you depend on this crate, you’ll need to migrate to Phantom or pin older Poulpy versions.

# FHE-RAM: Encrypted Random Access Memory

This repository demonstrates how [Poulpy](https://github.com/phantomzone-org/poulpy), a modular and high-performance lattice-based FHE library, can be used to implement fully homomorphic encrypted read/write RAM.

The FHE-RAM example uses a combination of GLWE, GGLWE and GGSW ciphertext operations to provide encrypted read/write access on an encrypted database using encrypted address.

## Parameterization & Performance

```rust
const LOG_N: usize = 12;                    // Lattice degree
const BASEK: usize = 17;                    // Torus 2^{-k} decomposition basis.
const RANK: usize = 1;                      // GLWE/GGLWE/GGSW rank.
const K_PT: usize = (u8::BITS as usize) + 1;// Ram plaintext (GLWE) Torus precision.
const K_CT: usize = BASEK * 3;              // Ram ciphertext (GLWE) Torus precision.
const K_ADDR: usize = BASEK * 4;            // Ram address (GGSW) Torus precision.
const K_EVK: usize = BASEK * 5;             // Ram evaluation keys (GGLWE) Torus precision
const XS: f64 = 0.5;                        // Secret-key distribution.
const XE: f64 = 3.2;                        // Noise standard deviation.
const DECOMP_N: [u8; 4] = [3, 3, 3, 3];     // Digit decomposition of N.
```

### Performance

```rust
const WORDSIZE: usize = 4;
const MAX_ADDR: usize = 1 << 18;
```

For a RAM-size of 2^18 entries, with each entry a 4xu8 word (i.e. RAM size is 1MB), the above parameterization enables 450ms read and 1200ms write (i9-12900K single thread) with at least ~40mio read/write without having to refresh the RAM.

### Security

The above paramterization offers ~168 bits of security. 

```
from estimator import *
n = 1<<12
q = 1<<85 # BASEK * 5
Xs = ND.SparseTernary(n=n, p=int(n/2)) # Hamming weight of 0.5 * n
Xe = ND.DiscreteGaussian(3.2)

# LWE
lwe = LWE.Parameters(n=n, q=q, Xs=Xs, Xe=Xe, m = 2*n)
lwe_res = LWE.estimate(lwe, red_cost_model = RC.BDGL16)

print("LWE Security")
print(lwe)

bkw                  :: rop: ≈2^619.3, m: ≈2^601.2, mem: ≈2^602.2, b: 7, t1: 7, t2: 143, ℓ: 6, #cod: ≈2^11.9, #top: 0, #test: 203, tag: coded-bkw
usvp                 :: rop: ≈2^169.6, red: ≈2^169.6, δ: 1.003557, β: 470, d: 7858, tag: usvp
bdd                  :: rop: ≈2^168.5, red: ≈2^168.5, svp: ≈2^163.9, β: 466, η: 505, d: 8078, tag: bdd
dual                 :: rop: ≈2^170.5, mem: ≈2^107.0, m: ≈2^12.0, β: 473, d: 8181, ↻: 1, tag: dual
dual_hybrid          :: rop: ≈2^167.9, red: ≈2^167.9, guess: ≈2^158.4, β: 464, p: 5, ζ: 20, t: 40, β': 464, N: ≈2^95.6, m: ≈2^12.0
```

## Installing Dependencies

This example requires a local installation of [Poulpy](https://github.com/phantomzone-org/poulpy).

1. Clone Poulpy into the parent directory:

```bash
git clone https://github.com/phantomzone-org/poulpy ../poulpy
```

2. Navigate to poulpy and initialise [spqlios-arithmetic](https://github.com/phantomzone-org/spqlios-arithmetic) as a submodule inside `backend`:

```bash
cd ../poulpy
git submodule update --init --recursive
```

3. Build [spqlios-arithmetic](https://github.com/phantomzone-org/spqlios-arithmetic):

On linux:

```bash
# at parent directory
cd ./poulpy/backend/spqlios-arithmetic
mkdir build && cd build
cmake .. -DENABLE_TESTING=off
make -j
cd ../../../../ # back to parent directory
```

On macos:

```bash
# at parent directory
cd ./poulpy/backend/spqlios-arithmetic
mkdir build && cd build
cmake .. -DENABLE_TESTING=off -DCMAKE_EXE_LINKER_FLAGS="-static-libstdc++"
make -j

cd spqlios && rm -rf *.dylib* && cd ..
cd ../../../../ # back to parent directory
```

## Running the Example

Run the RAM simulation example with:

```bash
cargo run --release --example fhe-ram
```

### Available API

```rust
// Global public parameters (e.g. cryptographic parameters)
let params = Parameters::new();

// Word-size, i.e. how many chunks of K_PT bits a word is made of.
// By default WORDSIZE=4 chunks of K_PT=8 bits, i.e. 32bit words.
let ws = params.word_size();
    
// Maximum supported address. In the default parameterization MAX_ADDR=1<<18;
// Each entry (address) stores WORDSIZE * K_PT bits.
let max_addr = params.max_addr();

// Generates a new secret along with the (public) evaluation key (evk).
// The evaluation key comprises log(N) automorphism keys (GGLWE) as well
// as the tensor key (rank choice 2 GGLWE), a.k.a relinearization key.
let (sk, evk) = gen_keys(&params);

// Create a new FHE-RAM instance.
let mut ram = Ram::new();

// Encrypt an array of bytes of length WORDSIZE * MAX_ADDR as vector of GLWE.
ram.encrypt_sk(&data, &sk);

// Allocates a new encrypted address (matrix of GGSW).
let mut addr: Address = Address::alloc(&params);

// Encrypt an address value.
addr.encrypt_sk(&params, idx, &sk);

// Read from the encrypted RAM at the encrypted address.
// Returns a vector of GLWE of length WORDSIZE.
let ct = ram.read(&addr, &keys);

// Same as read, but prepares the state for a subsequent write.
let ct = ram.read_prepare_write(&addr, &keys);

// Writes encrypted bytes to the encrypted RAM at the encrypted address.
// Takes as input a vector of GLWE of length WORDSIZE.
ram.write(&ct_w, &addr, &keys); 
```

## Disclaimer

This is **research code**, not production software. It is intended for experimentation and validation of encrypted memory concepts.

## License

Licensed under the [Apache License, Version 2.0](LICENSE).

## Citing

```
@misc{fhe-ram,
    title = {FHE-RAM},
    howpublished = {Online: \url{https://github.com/phantomzone-org/fhe-ram}},
    month = May,
    year = 2025,
    note = {Jean-Philippe Bossuat, Janmajaya Mall}
}
```
