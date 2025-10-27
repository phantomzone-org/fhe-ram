pub mod address;
pub mod base;
pub mod coordinate;
pub mod coordinate_prepared;
pub mod keys;
pub mod parameters;
pub mod ram;

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
