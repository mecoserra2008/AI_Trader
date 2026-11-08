//! Columnar batch builder used by every connector. Columns are pushed whole (no per-row maps),
//! which keeps millions of trades per day cheap.

use std::collections::BTreeMap;

use crate::schema::ColType;

#[derive(Clone, Debug)]
pub enum Col {
    Utf8(Vec<Option<String>>),
    F64(Vec<Option<f64>>),
    F32(Vec<Option<f32>>),
    I64(Vec<Option<i64>>),
    I32(Vec<Option<i32>>),
    Bool(Vec<Option<bool>>),
    TsMs(Vec<Option<i64>>),
}

impl Col {
    pub fn len(&self) -> usize {
        match self {
            Col::Utf8(v) => v.len(),
            Col::F64(v) => v.len(),
            Col::F32(v) => v.len(),
            Col::I64(v) => v.len(),
            Col::I32(v) => v.len(),
            Col::Bool(v) => v.len(),
            Col::TsMs(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn ty(&self) -> ColType {
        match self {
            Col::Utf8(_) => ColType::Utf8,
            Col::F64(_) => ColType::F64,
            Col::F32(_) => ColType::F32,
            Col::I64(_) => ColType::I64,
            Col::I32(_) => ColType::I32,
            Col::Bool(_) => ColType::Bool,
            Col::TsMs(_) => ColType::TsMs,
        }
    }

    pub fn null_count(&self) -> usize {
        fn n<T>(v: &[Option<T>]) -> usize {
            v.iter().filter(|x| x.is_none()).count()
        }
        match self {
            Col::Utf8(v) => n(v),
            Col::F64(v) => n(v),
            Col::F32(v) => n(v),
            Col::I64(v) => n(v),
            Col::I32(v) => n(v),
            Col::Bool(v) => n(v),
            Col::TsMs(v) => n(v),
        }
    }

    /// String form of one value, for partition paths.
    pub fn value_string(&self, i: usize) -> Option<String> {
        match self {
            Col::Utf8(v) => v[i].clone(),
            Col::I64(v) => v[i].map(|x| x.to_string()),
            Col::I32(v) => v[i].map(|x| x.to_string()),
            Col::Bool(v) => v[i].map(|x| x.to_string()),
            Col::TsMs(v) => v[i].map(|x| x.to_string()),
            Col::F64(v) => v[i].map(|x| x.to_string()),
            Col::F32(v) => v[i].map(|x| x.to_string()),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Batch {
    pub(crate) cols: BTreeMap<String, Col>,
}

impl Batch {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.cols.values().next().map(Col::len).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn col(&self, name: &str) -> Option<&Col> {
        self.cols.get(name)
    }

    pub fn push(mut self, name: &str, col: Col) -> Self {
        self.cols.insert(name.to_string(), col);
        self
    }

    pub fn utf8(self, name: &str, v: Vec<String>) -> Self {
        self.push(name, Col::Utf8(v.into_iter().map(Some).collect()))
    }

    pub fn utf8_opt(self, name: &str, v: Vec<Option<String>>) -> Self {
        self.push(name, Col::Utf8(v))
    }

    pub fn utf8_const(self, name: &str, value: &str, n: usize) -> Self {
        self.push(name, Col::Utf8(vec![Some(value.to_string()); n]))
    }

    pub fn f64(self, name: &str, v: Vec<f64>) -> Self {
        self.push(name, Col::F64(v.into_iter().map(Some).collect()))
    }

    pub fn f64_opt(self, name: &str, v: Vec<Option<f64>>) -> Self {
        self.push(name, Col::F64(v))
    }

    pub fn f32(self, name: &str, v: Vec<f32>) -> Self {
        self.push(name, Col::F32(v.into_iter().map(Some).collect()))
    }

    pub fn f32_opt(self, name: &str, v: Vec<Option<f32>>) -> Self {
        self.push(name, Col::F32(v))
    }

    pub fn i64(self, name: &str, v: Vec<i64>) -> Self {
        self.push(name, Col::I64(v.into_iter().map(Some).collect()))
    }

    pub fn i32(self, name: &str, v: Vec<i32>) -> Self {
        self.push(name, Col::I32(v.into_iter().map(Some).collect()))
    }

    pub fn i32_opt(self, name: &str, v: Vec<Option<i32>>) -> Self {
        self.push(name, Col::I32(v))
    }

    pub fn ts(self, name: &str, v: Vec<i64>) -> Self {
        self.push(name, Col::TsMs(v.into_iter().map(Some).collect()))
    }

    pub fn ts_opt(self, name: &str, v: Vec<Option<i64>>) -> Self {
        self.push(name, Col::TsMs(v))
    }
}

impl Col {
    fn extend(&mut self, other: Col) -> Result<(), Col> {
        match (self, other) {
            (Col::Utf8(a), Col::Utf8(b)) => a.extend(b),
            (Col::F64(a), Col::F64(b)) => a.extend(b),
            (Col::F32(a), Col::F32(b)) => a.extend(b),
            (Col::I64(a), Col::I64(b)) => a.extend(b),
            (Col::I32(a), Col::I32(b)) => a.extend(b),
            (Col::Bool(a), Col::Bool(b)) => a.extend(b),
            (Col::TsMs(a), Col::TsMs(b)) => a.extend(b),
            (_, b) => return Err(b),
        }
        Ok(())
    }
}

impl Batch {
    /// Append the rows of `other` (same column names and types) so several small batches become
    /// one write (one file per partition). An empty side is replaced by the other.
    pub fn append(&mut self, other: Batch) -> anyhow::Result<()> {
        if other.is_empty() {
            return Ok(());
        }
        if self.is_empty() {
            *self = other;
            return Ok(());
        }
        anyhow::ensure!(
            self.cols.len() == other.cols.len() && self.cols.keys().all(|k| other.cols.contains_key(k)),
            "batch append: column sets differ"
        );
        for (name, col) in &other.cols {
            let want = self.cols[name].ty();
            anyhow::ensure!(want == col.ty(), "batch append: column {name} is {want:?}, got {:?}", col.ty());
        }
        for (name, col) in other.cols {
            if let Some(mine) = self.cols.get_mut(&name) {
                mine.extend(col).map_err(|_| anyhow::anyhow!("batch append: type mismatch in {name}"))?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn append_concatenates_and_checks_columns() {
        let mut a = Batch::new().utf8_const("s", "x", 2).ts("t", vec![1, 2]);
        a.append(Batch::new().utf8_const("s", "y", 1).ts("t", vec![3])).unwrap();
        assert_eq!(a.len(), 3);
        assert_eq!(a.col("s").unwrap().value_string(2).as_deref(), Some("y"));
        assert_eq!(a.col("t").unwrap().value_string(2).as_deref(), Some("3"));
        let mut e = Batch::new();
        e.append(a.clone()).unwrap();
        assert_eq!(e.len(), 3);
        a.append(Batch::new()).unwrap();
        assert_eq!(a.len(), 3);
        assert!(a.clone().append(Batch::new().utf8_const("s", "z", 1)).is_err());
        assert!(a.append(Batch::new().utf8_const("s", "z", 1).i64("t", vec![4])).is_err());
        assert_eq!(a.col("s").unwrap().len(), 3, "failed append must not modify the batch");
    }
}
