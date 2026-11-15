//! A small deterministic multiplicative hasher for integer keys (order ids). std's SipHash is DoS-resistant but
//! several times slower; inputs here are our own files, and determinism matters more than hash flooding.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

#[derive(Default, Clone, Copy)]
pub struct IntHasher(u64);

impl Hasher for IntHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0.rotate_left(5) ^ b as u64).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
        }
    }

    #[inline]
    fn write_u64(&mut self, n: u64) {
        // Fibonacci hashing of the key, folded so the low bits used by the table are well mixed.
        let h = (self.0 ^ n).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        self.0 = h ^ (h >> 29);
    }

    #[inline]
    fn write_u32(&mut self, n: u32) {
        self.write_u64(n as u64)
    }
}

pub type IntMap<K, V> = HashMap<K, V, BuildHasherDefault<IntHasher>>;
