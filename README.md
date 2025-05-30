# FHE-RAM: Encrypted Random Access Memory Demonstration

**FHE-RAM** is an experimental prototype demonstrating how to implement fully homomorphic encrypted read/write RAM using [**Poulpy**](https://github.com/phantomzone-org/poulpy) — a modular and high-performance backend for lattice-based FHE schemes.

This project is **not** intended as a general-purpose library. It serves as a **research-oriented example** to highlight Poulpy's core features and APIs for secure, encrypted computation.

---

## 🧠 Purpose

This repository demonstrates how [Poulpy](https://github.com/phantomzone-org/poulpy) can be used to:

* Implement encrypted memory with read/write access.
* Leverage GLWE/GGLWE/GGSW ciphertexts for indexing and control logic.
* Perform non-trivial, stateful homomorphic operations efficiently.

---

## 🧱 Built With

* [Poulpy](https://github.com/phantomzone-org/poulpy): The core lattice-FHE backend.
* Rust: For performance and memory safety in low-level cryptographic code.

---

## 🔍 Capabilities

* Simulates a RAM with encrypted address and data lines.
* Supports homomorphic read and write operations.

---

## 📁 Repository Structure

```text
fhe-ram/
├── src/          # Core logic using Poulpy APIs
├── examples/     # Demonstration example(s)
├── Cargo.toml    # Project configuration
└── README.md     # You're here
```

---

## 🧰 Installing Dependencies

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

## ❗ Disclaimer

This is **research code**, not production software. It is intended for experimentation and validation of encrypted memory concepts.

---

## 📜 License

Licensed under the [Apache License, Version 2.0](LICENSE).

---

## 👥 Acknowledgments

This project is part of the [PhantomZone](https://github.com/phantomzone-org) initiative to advance homomorphic encryption and verifiable computing.

---
