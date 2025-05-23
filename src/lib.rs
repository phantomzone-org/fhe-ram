pub mod address;
pub(crate) mod packing;
pub mod ram;

#[cfg(test)]
mod test_fft64;

#[inline(always)]
pub fn reverse_bits_msb(x: usize, n: u32) -> usize {
    x.reverse_bits() >> (usize::BITS - n)
}
