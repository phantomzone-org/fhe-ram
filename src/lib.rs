pub mod address;
pub mod keys;
pub mod parameters;
pub mod ram;

// Backend selection based on target architecture
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub use poulpy_backend::FFT64Avx as BackendImpl;

#[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
pub use poulpy_backend::FFT64Ref as BackendImpl;

#[inline(always)]
pub fn reverse_bits_msb(x: usize, n: u32) -> usize {
    x.reverse_bits() >> (usize::BITS - n)
}
