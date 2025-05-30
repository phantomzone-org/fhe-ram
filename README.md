# FHE-RAM: Encrypted Random Access Memory Demonstration

This repository demonstrates how [Poulpy](https://github.com/phantomzone-org/poulpy), a modular and high-performance lattice-based FHE library, can be used to implement fully homomorphic encrypted read/write RAM.

The FHE-RAM example use a combination of GLWE, GGLWE and GGSW ciphertext operations to provide encrypted read/write access on an encrypted database using encrypted address.

---

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

For a RAM-size of 2^18 with 4xu8 words, the above parameterization enables 500ms read and 1500ms write (i9-12900K single thread) with at least ~40mio read/write without having to refresh the RAM.

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


RAM Size| Word Size | Age | Country    |
|-------|-----------|-----|------------|
  2^14  |   4xu8    |     |            |
  2^16  |   4xu8    | 30  | USA        |
  2^18  |   8xu8    | 25  | Canada     |
  2^20  |           | 35  | Australia  |

---

## Repository Structure

```text
fhe-ram/
├── src/          # Core logic using Poulpy APIs
├── examples/     # Demonstration example(s)
├── Cargo.toml    # Project configuration
└── README.md     # You're here
```

---

## Installing Dependencies

This example requires a local installation of [Poulpy](https://github.com/phantomzone-org/poulpy).

1. Clone Poulpy into the parent directory:

```bash
git clone https://github.com/phantomzone-org/poulpy ../poulpy
```

2. Build the [spqlios-arithmetic](https://github.com/phantomzone-org/spqlios-arithmetic) backend:

```bash
cd ../poulpy/backend/spqlios-arithmetic
mkdir build && cd build
cmake ..
make
```

> ⚠️ These steps assume a Linux environment. For other platforms, refer to the [spqlios-arithmetic build guide](https://github.com/tfhe/spqlios-arithmetic/wiki/build).

---

## ▶️ Running the Example

Run the RAM simulation example with:

```bash
cargo run --release --example fhe-ram
```

---

## Disclaimer

This is **research code**, not production software. It is intended for experimentation and validation of encrypted memory concepts.

---

## License

Licensed under the [Apache License, Version 2.0](LICENSE).

---

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