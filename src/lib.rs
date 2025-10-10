mod address;
mod base;
mod coordinate;
mod coordinate_prepared;
mod keys;
mod parameters;
mod ram;

pub use address::*;
pub(crate) use base::*;
pub(crate) use coordinate::*;
pub(crate) use coordinate_prepared::*;
pub use keys::*;
pub use parameters::*;
pub use ram::*;

#[inline(always)]
pub fn reverse_bits_msb(x: usize, n: u32) -> usize {
    x.reverse_bits() >> (usize::BITS - n)
}
