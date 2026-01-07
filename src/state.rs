/// ==============================================================================================
/// =============================== Single Spatiotemporal State ==================================
/// ==============================================================================================

use serde::{Deserialize, Serialize};
use num_traits::Float;
use ndarray::parallel::prelude::*;
use ndarray::{Array1, ArrayD, IxDyn};

/// Representation mode:
///     Two mutually exclusive conventions for what `state` means:
///         - `Frequency`: entries live on the simplex (mass = 1),
///         - `Population`: entries carry absolute counts (mass not necessarily 1)
///     where 'cutoff' is an absorbing boundary.

#[derive(Clone, Serialize, Deserialize)]
pub enum Mode<T> {
    Frequency { cutoff: T },
    Population { cutoff: T, carrying_capacity: Option<T> },
}

/// Snapshot at an integer time index.
///     - `state`: well-mixed / global vector (d)
///     - `space`: spatial field (shape arbitrary: [X, Y, Z, ...])

#[derive(Clone, Serialize, Deserialize)]
pub struct GeneticState<T> {
    pub mode: Mode<T>,
    pub time: usize,
    pub state: Array1<T>,
    pub space: Option<ArrayD<T>>,
    pub mass: T,
}

/// Constructors
impl<T> GeneticState<T>
where
    T: Float + Clone + Default + std::iter::Sum<T>,
{
    #[inline]
    pub fn from_arrays(
        mode: Mode<T>,
        time: usize,
        mut state: Array1<T>,
        space: Option<ArrayD<T>>,
    ) -> Self {
        let sum: T = state.iter().copied().sum();

        let mass = match &mode {
            Mode::Frequency { .. } => {
                if sum > T::zero() {
                    let inv = T::one() / sum;
                    state.iter_mut().for_each(|x| *x = *x * inv);
                    T::one()
                } else {
                    let d = state.len();
                    if d > 0 {
                        let v = T::one() / T::from(d).unwrap();
                        state.iter_mut().for_each(|x| *x = v);
                        T::one()
                    } else {
                        T::zero()
                    }
                }
            }
            Mode::Population { .. } => sum,
        };

        Self { mode, time, state, space, mass }
    }

    #[inline]
    pub fn empty(
        mode: Mode<T>,
        time: usize,
        num_taxa: usize,
        space_shape: Option<&[usize]>,
    ) -> Self {
        let state = Array1::from_elem(num_taxa, T::default());
        let space = match space_shape {
            Some(shape) => Some(ArrayD::from_elem(IxDyn(shape), T::default())),
            None => None,
        };

        Self::from_arrays(mode, time, state, space)
    }
}


/// ==============================================================================================
/// =================================== State Sanitization =======================================
/// ==============================================================================================
impl<T> GeneticState<T>
where
    T: Float + Send + Sync + std::iter::Sum<T>,
{
    // Hard-threshold invalid / nonpositive / below-cutoff entries to `zero`.
    //     - `cutoff` is assumed nonnegative by the caller
    //     - `zero` is carried to avoid repeated `T::zero()` calls
    #[inline]
    fn apply_cutoff(&mut self, cutoff: T, zero: T) {
        self.state.par_iter_mut().for_each(|x| {
            if !x.is_finite() || *x <= zero || *x < cutoff {
                *x = zero;
            }
        });
    }

    // Enforce mode-specific invariants and update `mass`.
    //     - `Frequency`: project onto simplex (sum = 1), set `mass = 1`
    //     - `Population`: apply cutoff, optionally cap at `carrying_capacity`, set `mass ≈ round(sum)`
    #[inline]
    pub fn sanitize(&mut self) {
        let zero = T::zero();

        match self.mode {
            Mode::Frequency { cutoff } => {
                // Cutoff sanitize.
                let cutoff = cutoff.max(zero);
                self.apply_cutoff(cutoff, zero);

                // Renormalize onto simplex; fallback to uniform if all mass removed.
                let sum: T = self.state.par_iter().copied().sum();
                if sum > zero {
                    let inv = T::one() / sum;
                    self.state.par_iter_mut().for_each(|x| *x = *x * inv);
                } else {
                    let d = self.state.len();
                    if d > 0 {
                        let v = T::one() / T::from(d).unwrap();
                        self.state.par_iter_mut().for_each(|x| *x = v);
                    }
                }

                self.mass = T::one(); // simplex convention
            }
            Mode::Population {
                cutoff,
                carrying_capacity,
            } => {
                // Cutoff sanitize.
                let cutoff = cutoff.max(zero);
                self.apply_cutoff(cutoff, zero);

                let sum: T = self.state.par_iter().copied().sum();

                // No capacity constraint: just report (rounded) mass.
                let Some(capacity) = carrying_capacity else {
                    self.mass = sum.round().max(zero);
                    return;
                };

                // Degenerate capacity: zero out.
                if capacity <= zero {
                    self.state.par_iter_mut().for_each(|x| *x = zero);
                    self.mass = zero;
                    return;
                }

                // Under capacity: keep as-is.
                if sum < capacity {
                    self.mass = sum.round().max(zero);
                    return;
                }

                // Exactly at capacity: keep as-is (avoid divide-by-zero concerns elsewhere).
                if sum == capacity {
                    self.mass = capacity.round().max(zero);
                    return;
                }

                // Over capacity: rescale down to hit the cap exactly.
                let scale = capacity / sum;
                self.state.par_iter_mut().for_each(|x| *x = *x * scale);

                self.mass = capacity.round().max(zero);
            }
        }
    }
}


/// ==============================================================================================
/// ================================= Time Series Container ======================================
/// ==============================================================================================
use std::fs::{create_dir_all, File};
use std::io::{Error, ErrorKind, Result, Write};
use std::path::Path;

/// Generic time series dataframe for genetic states.
///     Design notes:
///         - `epoch` is the index of the dataframe
///         - per-sample time lives inside `GeneticState::time`

#[derive(Clone, Serialize)]
pub struct GeneticStateTimeSeries<'a, T> {
    pub epoch: usize,     // Epoch index
    pub states: Vec<&'a T>, // Ordered samples (borrowed)
}

impl<'a, T> GeneticStateTimeSeries<'a, T> {
    /// Empty time series (no samples yet).
    #[inline]
    pub fn empty(epoch: usize) -> Self {
        Self {
            epoch,
            states: Vec::new(),
        }
    }

    /// Add a borrowed state (no copy).
    #[inline]
    pub fn add(&mut self, state: &'a T) {
        self.states.push(state);
    }
}

impl<'a, T> GeneticStateTimeSeries<'a, GeneticState<T>>
where
    T: Serialize,
{
    /// Write the list of `GeneticState` into `{output_path}/{epoch}.json` (pretty-printed).
    pub fn save(&self, output_path: &Path) -> Result<()> {
        create_dir_all(output_path).map_err(|e| {
            Error::new(
                e.kind(),
                format!("GSTS::save: create dir {}: {e}", output_path.display()),
            )
        })?;

        let file_path = output_path.join(format!("{}.json", self.epoch));
        let json = serde_json::to_string_pretty(&self.states).map_err(|e| {
            Error::new(
                ErrorKind::InvalidData,
                format!("GSTS::save: serialize {}: {e}", file_path.display()),
            )
        })?;

        let mut file = File::create(&file_path).map_err(|e| {
            Error::new(
                e.kind(),
                format!("GSTS::save: create {}: {e}", file_path.display()),
            )
        })?;

        file.write_all(json.as_bytes()).map_err(|e| {
            Error::new(
                e.kind(),
                format!("GSTS::save: write {}: {e}", file_path.display()),
            )
        })?;

        Ok(())
    }
}
