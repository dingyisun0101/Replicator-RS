//! =============================================================================================
//! Simplex: probability-vector wrapper + feasibility restoration + simplex-native noise models
//! =============================================================================================
//!
//! This module centralizes logic for working on the probability simplex:
//!     - Nonnegative entries: ν_i >= 0
//!     - Unit mass: Σ_i ν_i = 1
//!
//! It provides:
//!     1) `Simplex`: a thin wrapper around `ndarray::Array1<f64>` with a stored cutoff policy
//!     2) `Noise` / `NoiseKind`: optional stochasticity applied to a `Simplex`
//!     3) `NoiseContext`: reusable workspace to avoid per-step allocations
//!     4) `apply_noise_inplace`: applies noise then calls `sanitize()`
//!
//! DESIGN NOTES
//! -----------
//! - We do not enforce invariants via type tricks; instead, we provide a canonical `sanitize()`
//!   routine and call it after any operation that can violate feasibility.
//! - `cutoff` is an absorbing boundary:
//!     - entries < cutoff are hard-zeroed before renormalization
//!     - negative cutoff is treated as 0
//!
//! =============================================================================================
#![allow(dead_code)]

use ndarray::Array1;
use crate::taxon_table::TaxonTable;

/// ==============================================================================================
/// ======================== Simplex type for probability vectors ================================
/// ==============================================================================================

/// A thin, opinionated wrapper around an `ndarray::Array1<f64>` intended to represent a
/// probability simplex point:
///     - Nonnegative entries: ν_i >= 0
///     - Unit mass: Σ_i ν_i = 1
///
/// Centralizes:
///     1) Sanitizing/renormalizing onto the simplex (with a hard cutoff)
///     2) Writing frequencies into `TaxonTable<f64>` with guaranteed on-simplex output


#[derive(Clone, Debug)]
pub struct Simplex {
    data: Array1<f64>,
    cutoff: f64,
}

impl Simplex {
    /// Construct from an owned vector and immediately sanitize using the stored `cutoff`.
    #[inline]
    pub fn from_vec(v: Vec<f64>, cutoff: f64) -> Self {
        debug_assert!(!v.is_empty(), "Simplex::from_vec: empty vector");
        let mut s = Self {
            data: Array1::from_vec(v),
            cutoff: cutoff.max(0.0),
        };
        s.sanitize();
        s
    }

    /// Construct from a borrowed slice (copies) and immediately sanitize using the stored `cutoff`.
    #[inline]
    pub fn from_slice(slice: &[f64], cutoff: f64) -> Self {
        debug_assert!(!slice.is_empty(), "Simplex::from_slice: empty slice");
        Self::from_vec(slice.to_vec(), cutoff)
    }

    /// Construct the uniform simplex point of dim `d`: ν_i = 1/d for all i.
    #[inline]
    pub fn uniform(d: usize, cutoff: f64) -> Self {
        assert!(d > 0, "Simplex::uniform: dim must be > 0");
        let mut s = Self {
            data: Array1::from_elem(d, 1.0 / (d as f64)),
            cutoff: cutoff.max(0.0),
        };
        s.sanitize();
        s
    }

    /// Dimension of the simplex (number of taxa).
    #[inline]
    pub fn dim(&self) -> usize {
        self.data.len()
    }

    /// The internally stored absorbing-boundary cutoff.
    #[inline]
    pub fn cutoff(&self) -> f64 {
        self.cutoff
    }

    /// Update the cutoff policy and re-sanitize under the new threshold.
    #[inline]
    pub fn set_cutoff(&mut self, cutoff: f64) {
        self.cutoff = cutoff.max(0.0);
        self.sanitize();
    }

    /// Borrow as an `ndarray` view for solver math.
    #[inline]
    pub fn as_array(&self) -> &Array1<f64> {
        &self.data
    }

    /// Mutable borrow as an `ndarray` view for in-place solver updates.
    ///
    /// Important: mutating through this view can violate simplex constraints.
    /// Call `sanitize()` after any operation that may produce negatives, NaNs, or mass drift.
    #[inline]
    pub fn as_array_mut(&mut self) -> &mut Array1<f64> {
        &mut self.data
    }

    /// Canonical projection onto the simplex using the internally stored cutoff.
    ///
    /// Steps:
    ///     1) Non-finite or non-positive entries → 0
    ///     2) Entries `< cutoff` → 0
    ///     3) Renormalize to sum=1; if everything becomes zero, fall back to uniform
    #[inline]
    pub fn sanitize(&mut self) {
        debug_assert!(self.dim() > 0);
        let thresh = self.cutoff;

        // --------------------------------------------------------------------------------------
        // (1) Clamp invalid values and apply the absorbing boundary
        // --------------------------------------------------------------------------------------
        for x in self.data.iter_mut() {
            if !x.is_finite() || *x <= 0.0 || *x < thresh {
                *x = 0.0;
            }
        }

        // --------------------------------------------------------------------------------------
        // (2) Renormalize; if all mass vanished, fall back to uniform
        // --------------------------------------------------------------------------------------
        let s: f64 = self.data.iter().copied().sum();
        if s > 0.0 {
            for x in self.data.iter_mut() {
                *x /= s;
            }
        } else {
            let c = 1.0 / (self.dim() as f64);
            for x in self.data.iter_mut() {
                *x = c;
            }
        }
    }

    /// Copy this simplex point into a `TaxonTable<f64>` and advance the table time (`tick()`).
    ///
    /// Defensive behavior:
    ///     - Re-sanitizes a temporary copy using the same cutoff policy before writing,
    ///       guaranteeing persisted JSON remains a valid simplex point.
    #[inline]
    pub fn write_into_table(&self, table: &mut TaxonTable<f64>) {
        let d = table.data.len();
        assert_eq!(
            self.dim(),
            d,
            "Simplex::write_into_table: length mismatch (simplex={}, table={})",
            self.dim(),
            d
        );

        let mut tmp = Array1::from_vec(self.data.to_vec());
        {
            let cutoff_nonneg = self.cutoff;

            // clamp + cutoff
            for x in tmp.iter_mut() {
                if !x.is_finite() || *x <= 0.0 || *x < cutoff_nonneg {
                    *x = 0.0;
                }
            }

            // renormalize or uniform fallback
            let s: f64 = tmp.iter().copied().sum();
            if s > 0.0 {
                for x in tmp.iter_mut() {
                    *x /= s;
                }
            } else {
                let c = 1.0 / (tmp.len() as f64);
                for x in tmp.iter_mut() {
                    *x = c;
                }
            }
        }

        table.data.copy_from_slice(tmp.as_slice().expect("Simplex: contiguous"));
        table.tick();
    }
}
