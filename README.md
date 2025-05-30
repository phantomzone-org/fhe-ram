# FHE-RAM: Encrypted Random Access Memory Demonstration

This repository demonstrates how [Poulpy](https://github.com/phantomzone-org/poulpy), a modular and high-performance lattice-based FHE library, can be used to implement fully homomorphic encrypted read/write RAM.

The FHE-RAM example use a combination of GLWE, GGLWE and GGSW ciphertext operations to provide encrypted read/write access on an encrypted database using encrypted address.

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