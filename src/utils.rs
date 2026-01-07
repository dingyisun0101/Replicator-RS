// src/utils.rs
use ndarray::Array1;
use num_traits::Float;

use crate::state::{GeneticState, Mode};

/// ==============================================================================================
/// ============================= Constructors / Initialization Helpers ===========================
/// ==============================================================================================

/// Create a well-mixed (spatially uniform / no-grid) `GeneticState<T>` at time 0.
///     Behavior:
///         - `Mode::Frequency`: initialize uniform simplex (sum = 1)
///         - `Mode::Population`: initialize equal counts per taxon (sum = num_taxa * per_taxon)
/// Notes:
///     - Does not allocate any spatial field (`space = None`)
///     - Does not call `sanitize()`; caller may do so if invariants depend on cutoff/capacity
pub fn create_well_mixed_gs<T>(
    mode: Mode<T>,             // representation convention (frequency vs population)
    num_taxa: usize,           // dimensionality d
    population_i: Option<T>,   // per-taxon population (only used for Population mode)
) -> GeneticState<T>
where
    T: Float + Clone + Default + std::iter::Sum<T>,
{
    let zero = T::zero();
    let mut state = Array1::from_elem(num_taxa, zero);

    if num_taxa > 0 {
        match &mode {
            Mode::Frequency { .. } => {
                // Uniform simplex point: ν_i = 1/d.
                let v = T::one() / T::from(num_taxa).unwrap();
                state.iter_mut().for_each(|x| *x = v);
            }
            Mode::Population { .. } => {
                // Equal counts per taxon: n_i = population_i (default 1).
                let per_taxon = population_i.unwrap_or_else(T::one);
                state.iter_mut().for_each(|x| *x = per_taxon);
            }
        }
    }

    // time = 0, space = None (well-mixed)
    GeneticState::from_arrays(mode, 0, state, None)
}
