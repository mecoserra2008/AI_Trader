//! Exact decimal rounding to venue tick / lot steps. Floating-point division drifts
//! (0.3 / 0.1 = 2.9999…), which silently under-sizes or produces invalid order quantities.

use rust_decimal::Decimal;
use rust_decimal::prelude::{FromPrimitive, ToPrimitive};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Round {
    Down,
    Up,
    Nearest,
}

fn dec(x: f64) -> Decimal {
    Decimal::from_f64(x).unwrap_or_default().normalize()
}

/// Round `value` to an integer multiple of `step`.
pub fn round_to_step(value: f64, step: f64, mode: Round) -> f64 {
    if step <= 0.0 {
        return value;
    }
    let (v, s) = (dec(value), dec(step));
    let units = v / s;
    let units = match mode {
        Round::Down => units.floor(),
        Round::Up => units.ceil(),
        Round::Nearest => units.round(),
    };
    (units * s).to_f64().unwrap_or(f64::NAN)
}

/// Format a value already on the `step` grid with exactly the step's decimal places (for venue APIs).
pub fn format_step(value: f64, step: f64) -> String {
    let scale = dec(step).scale();
    let mut d = dec(value);
    d.rescale(scale);
    d.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounding_is_exact() {
        assert_eq!(round_to_step(0.3, 0.1, Round::Down), 0.3);
        assert_eq!(round_to_step(1.23456, 0.001, Round::Down), 1.234);
        assert_eq!(round_to_step(1.23401, 0.001, Round::Up), 1.235);
        assert_eq!(round_to_step(75812.27, 0.1, Round::Nearest), 75812.3);
        assert_eq!(format_step(0.5, 0.001), "0.500");
        assert_eq!(format_step(75812.3, 0.1), "75812.3");
        assert_eq!(format_step(3.0, 1.0), "3");
    }
}
