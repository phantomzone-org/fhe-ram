pub mod address;
pub mod keys;
pub mod parameters;
pub mod ram;

pub(crate) mod packing;

#[cfg(test)]
mod test_fft64;

#[inline(always)]
pub fn reverse_bits_msb(x: usize, n: u32) -> usize {
    x.reverse_bits() >> (usize::BITS - n)
}
