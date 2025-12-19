//! =============================================================================================
//! TaxonTable: lightweight time-stamped state container + JSON I/O + freq/pop conversions
//! =============================================================================================
//!
//! This module provides a small, generic `TaxonTable<T>` used to store either:
//!     - **Populations** (e.g. `usize` counts), or
//!     - **Frequencies** (e.g. `f64` simplex points, Σ=1).
//!
//! The design is intentionally minimal and "data-first":
//!     - `time: usize` is an integer clock advanced by `tick()`.
//!     - `data: Vec<T>` stores taxa values in a fixed ordering.
//!
//! REFACTOR HIGHLIGHTS
//! -------------------
//! - Unified documentation/comment style to match the replicator solver module.
//! - Kept core semantics unchanged, but tightened error messages and documentation.
//! - Clarified trait bounds: `Scalar` for numeric operations and `DeserializeOwned` for read.
//! - JSON I/O lives alongside the container, with explicit `save_to_json` / `read_from_json`.
//! - Frequency/population casting stays as specialized impl blocks on `TaxonTable<f64>` and
//!   `TaxonTable<usize>`.
//!
//! NOTES
//! -----
//! - `Scalar` is the PiP numeric trait used throughout your codebase; it provides:
//!     - `zero()`
//!     - `is_zero()`
//!     - arithmetic ops and `abs_real()`
//! - `l2_dist` uses rayon parallel iterators for performance on large vectors.
//!
//! =============================================================================================
#![allow(dead_code)]

use std::fs::File;
use std::io::{Error, ErrorKind, Result, Write};
use std::path::Path;

use num_traits::NumCast;
use physics_in_parallel::math::scalar::Scalar;
use rayon::prelude::*;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

/// ==============================================================================================
/// =============================== Model Information Tracking ===================================
/// ==============================================================================================

/// A time-stamped vector of taxa values.
///
/// Typical instantiations:
///     - `TaxonTable<usize>`: integer population counts
///     - `TaxonTable<f64>`: frequency vector (intended to lie on the simplex)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TaxonTable<T>
where
    T: Scalar,
{
    /// Discrete time index.
    pub time: usize,

    /// Per-taxon values (counts or frequencies depending on `T`).
    pub data: Vec<T>,
}

/// ==============================================================================================
/// ================================ Core utilities (generic) ====================================
/// ==============================================================================================

#[allow(dead_code)]
impl<T> TaxonTable<T>
where
    T: Scalar,
{
    /// Create a new table with `time=0` and `num_taxa` entries initialized to `T::zero()`.
    #[inline]
    pub fn new(num_taxa: usize) -> Self {
        Self {
            time: 0,
            data: vec![T::zero(); num_taxa],
        }
    }

    /// Sum of populations as `usize` (via `NumCast`; non-representable → 0).
    ///
    /// Intended use:
    ///     - primarily meaningful when `T` is an integer-like type
    ///     - will also "work" for floats by truncation via `NumCast`
    #[inline]
    pub fn total_population(&self) -> usize {
        self.data
            .iter()
            .copied()
            .map(|x| NumCast::from(x).unwrap_or(0usize))
            .sum()
    }

    /// Advance time by one. Returns `false` when only one (or zero) taxa remain nonzero.
    ///
    /// Uses `Scalar::is_zero()` so it works for ints/floats/complex.
    #[inline]
    pub fn tick(&mut self) -> bool {
        self.time += 1;
        let nonzero = self.data.iter().filter(|&c| !c.is_zero()).count();
        nonzero != 1
    }

    /// 2-norm of the difference in population vectors (returns `f64`).
    ///
    /// Works for any `T: Scalar` by reducing with `abs_real()` then casting to `f64`.
    pub fn l2_dist(&self, other: &TaxonTable<T>) -> f64 {
        self.data
            .par_iter()
            .zip(&other.data)
            .map(|(&a, &b)| {
                let dr = (a - b).abs_real();
                let x: f64 = NumCast::from(dr).unwrap_or(0.0);
                x * x
            })
            .sum::<f64>()
            .sqrt()
    }

    /// Indices of nonzero taxa.
    #[inline]
    pub fn live_taxa(&self) -> Vec<usize> {
        self.data
            .iter()
            .enumerate()
            .filter_map(|(i, c)| if !c.is_zero() { Some(i) } else { None })
            .collect()
    }
}

/// ==============================================================================================
/// ====================================== JSON I/O ==============================================
/// ==============================================================================================

#[allow(dead_code)]
impl<T> TaxonTable<T>
where
    T: Scalar + DeserializeOwned,
{
    /// Write this table to JSON at `output_path` (pretty-printed).
    pub fn save_to_json(&self, output_path: &Path) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        let mut file = File::create(output_path).map_err(|e| {
            Error::new(
                e.kind(),
                format!("TaxonTable::save_to_json: create {}: {e}", output_path.display()),
            )
        })?;
        file.write_all(json.as_bytes()).map_err(|e| {
            Error::new(
                e.kind(),
                format!("TaxonTable::save_to_json: write {}: {e}", output_path.display()),
            )
        })?;
        Ok(())
    }

    /// Read a `TaxonTable<T>` from an exact JSON file path.
    pub fn read_from_json(path: &Path) -> Result<Self> {
        let f = File::open(path).map_err(|e| {
            Error::new(
                e.kind(),
                format!("TaxonTable::read_from_json: open {}: {e}", path.display()),
            )
        })?;

        serde_json::from_reader::<_, Self>(f).map_err(|e| {
            Error::new(
                ErrorKind::InvalidData,
                format!("TaxonTable::read_from_json: parse {}: {e}", path.display()),
            )
        })
    }
}




/// ==============================================================================================
/// ======================== Type casting: Population <-> Frequency ==============================
/// ==============================================================================================
#[allow(dead_code)]
impl TaxonTable<f64> {
    /// Convert **frequencies** (Σ≈1) into **integer populations** that sum to `population`.
    ///
    /// Behavior:
    ///     - Clamps non-finite/negative entries to 0, then renormalizes with uniform fallback.
    ///     - Uses floor + largest-fractional-part correction to match the exact total.
    pub fn to_populations(&self, population: usize) -> TaxonTable<usize> {
        let l = self.data.len();

        // Edge cases
        if l == 0 {
            return TaxonTable::<usize> {
                time: self.time,
                data: Vec::new(),
            };
        }
        if population == 0 {
            return TaxonTable::<usize> {
                time: self.time,
                data: vec![0; l],
            };
        }

        // --------------------------------------------------------------------------------------
        // (1) Copy + clamp to nonnegative, drop non-finite; renormalize (uniform fallback).
        // --------------------------------------------------------------------------------------
        let mut freq = self.data.clone();
        for x in freq.iter_mut() {
            if !x.is_finite() || *x <= 0.0 {
                *x = 0.0;
            }
        }
        let s: f64 = freq.iter().copied().sum();
        if s > 0.0 {
            for x in freq.iter_mut() {
                *x /= s;
            }
        } else {
            let c = 1.0 / (l as f64);
            for x in freq.iter_mut() {
                *x = c;
            }
        }

        // --------------------------------------------------------------------------------------
        // (2) Raw expected counts and floor
        // --------------------------------------------------------------------------------------
        let raw: Vec<f64> = freq.iter().map(|&p| p * population as f64).collect();
        let mut counts: Vec<usize> = raw.iter().map(|&x| x.floor() as usize).collect();

        // --------------------------------------------------------------------------------------
        // (3) Adjust to exact total using largest remainder method
        // --------------------------------------------------------------------------------------
        let sum_now: usize = counts.iter().sum();
        let diff: isize = population as isize - sum_now as isize;

        if diff != 0 {
            let mut idx: Vec<usize> = (0..l).collect();
            idx.sort_by(|&a, &b| {
                let fa = raw[a] - counts[a] as f64;
                let fb = raw[b] - counts[b] as f64;
                fb.partial_cmp(&fa).unwrap() // descending by fractional part
            });

            if diff > 0 {
                for &i in idx.iter().take(diff as usize) {
                    counts[i] += 1;
                }
            } else {
                for &i in idx.iter().rev().take((-diff) as usize) {
                    if counts[i] > 0 {
                        counts[i] -= 1;
                    }
                }
            }
        }

        TaxonTable::<usize> {
            time: self.time,
            data: counts,
        }
    }
}

impl TaxonTable<usize> {
    /// Convert **integer populations** into **frequencies** on the simplex (Σ=1).
    ///
    /// Behavior:
    ///     - If the total is zero, returns a uniform distribution.
    pub fn to_frequencies(&self) -> TaxonTable<f64> {
        let l = self.data.len();

        if l == 0 {
            return TaxonTable::<f64> {
                time: self.time,
                data: Vec::new(),
            };
        }

        let total: usize = self.data.iter().sum();
        if total == 0 {
            let c = 1.0 / (l as f64);
            return TaxonTable::<f64> {
                time: self.time,
                data: vec![c; l],
            };
        }

        let total_f = total as f64;
        let freqs: Vec<f64> = self.data.iter().map(|&n| (n as f64) / total_f).collect();

        TaxonTable::<f64> {
            time: self.time,
            data: freqs,
        }
    }
}
