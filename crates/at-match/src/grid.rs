//! Exact tick / lot grid. A step such as 0.1 is held as mantissa x 10^-decimals (1 x 10^-1), so integer ticks
//! format back to the exact decimal string the venue prints, and off-grid inputs are detected instead of rounded.

use std::fmt::Write as _;

#[derive(Clone, Copy, Debug, PartialEq)]
struct Step {
    value: f64,
    mantissa: i64,
    decimals: u32,
    pow: f64,
}

impl Step {
    fn new(step: f64) -> Result<Step, String> {
        if !(step.is_finite() && step > 0.0) {
            return Err(format!("step must be finite and > 0, got {step}"));
        }
        // f64 Display is the shortest round-trip decimal and never uses an exponent.
        let s = format!("{step}");
        let (int, frac) = s.split_once('.').unwrap_or((&s, ""));
        let digits = format!("{int}{frac}");
        let mantissa: i64 = digits.trim_start_matches('0').parse().map_err(|_| format!("step {step} is not a plain decimal"))?;
        let decimals = frac.len() as u32;
        if decimals > 15 || mantissa <= 0 {
            return Err(format!("step {step} has too many decimals"));
        }
        Ok(Step { value: step, mantissa, decimals, pow: 10f64.powi(decimals as i32) })
    }

    /// Integer units of the step, or None if `x` is not finite or not on the grid (relative tolerance 1e-9).
    #[inline]
    fn units(&self, x: f64) -> Option<i64> {
        if !x.is_finite() {
            return None;
        }
        let u = x / self.value;
        let r = u.round();
        if (u - r).abs() <= 1e-9 * r.abs().max(1.0) && r.abs() < 9.0e15 { Some(r as i64) } else { None }
    }

    #[inline]
    fn scaled(&self, units: i64) -> f64 {
        (units as f64 * self.mantissa as f64) / self.pow
    }

    fn write(&self, units: i64, out: &mut String) {
        let v = units as i128 * self.mantissa as i128;
        if self.decimals == 0 {
            let _ = write!(out, "{v}");
            return;
        }
        let p = 10i128.pow(self.decimals);
        let (sign, a) = if v < 0 { ("-", -v) } else { ("", v) };
        let _ = write!(out, "{sign}{}.{:0width$}", a / p, a % p, width = self.decimals as usize);
    }
}

/// Price (tick) and quantity (lot) grid of one instrument.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Grid {
    tick: Step,
    lot: Step,
}

impl Grid {
    pub fn new(tick_size: f64, qty_step: f64) -> Result<Grid, String> {
        Ok(Grid { tick: Step::new(tick_size).map_err(|e| format!("tick_size: {e}"))?, lot: Step::new(qty_step).map_err(|e| format!("qty_step: {e}"))? })
    }

    pub fn tick_size(&self) -> f64 {
        self.tick.value
    }

    pub fn qty_step(&self) -> f64 {
        self.lot.value
    }

    /// Price -> ticks; None when off the tick grid or not finite.
    #[inline]
    pub fn ticks(&self, price: f64) -> Option<i64> {
        self.tick.units(price)
    }

    /// Qty -> lots; None when off the step grid or not finite.
    #[inline]
    pub fn lots(&self, qty: f64) -> Option<i64> {
        self.lot.units(qty)
    }

    /// Lots of a quantity rounded to the nearest step (for limits such as min_qty, which are on the grid).
    pub fn lots_nearest(&self, qty: f64) -> i64 {
        (qty / self.lot.value).round() as i64
    }

    #[inline]
    pub fn price(&self, ticks: i64) -> f64 {
        self.tick.scaled(ticks)
    }

    #[inline]
    pub fn qty(&self, lots: i64) -> f64 {
        self.lot.scaled(lots)
    }

    /// Exact decimal text of a price, with the tick's decimal places (Bybit style, e.g. "75812.3").
    pub fn write_price(&self, ticks: i64, out: &mut String) {
        self.tick.write(ticks, out)
    }

    pub fn write_qty(&self, lots: i64, out: &mut String) {
        self.lot.write(lots, out)
    }

    pub fn fmt_price(&self, ticks: i64) -> String {
        let mut s = String::new();
        self.write_price(ticks, &mut s);
        s
    }

    pub fn fmt_qty(&self, lots: i64) -> String {
        let mut s = String::new();
        self.write_qty(lots, &mut s);
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_is_exact() {
        let g = Grid::new(0.1, 0.001).unwrap();
        assert_eq!(g.ticks(75812.3), Some(758123));
        assert_eq!(g.ticks(75812.35), None, "off tick");
        assert_eq!(g.lots(0.3), Some(300));
        assert_eq!(g.lots(0.0005), None);
        assert_eq!(g.fmt_price(758123), "75812.3");
        assert_eq!(g.fmt_qty(1), "0.001");
        assert_eq!(g.fmt_qty(1500), "1.500");
        assert_eq!(g.price(758123), 75812.3);
        let h = Grid::new(0.5, 1.0).unwrap();
        assert_eq!(h.ticks(100.5), Some(201));
        assert_eq!(h.fmt_price(201), "100.5");
        assert_eq!(h.fmt_qty(7), "7");
        let k = Grid::new(0.0001, 10.0).unwrap();
        assert_eq!(k.fmt_price(12345), "1.2345");
        assert_eq!(k.fmt_qty(3), "30");
        assert_eq!(k.lots(25.0), None);
        assert!(Grid::new(0.0, 1.0).is_err());
        assert!(Grid::new(f64::NAN, 1.0).is_err());
    }
}
