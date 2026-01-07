use std::io::Result;
use std::path::Path;

use ndarray::{Array1, Array2};

use crate::state::Mode;
use crate::utils::create_well_mixed_gs;
use crate::solvers::non_spatial::noise::Noise;
use crate::solvers::non_spatial::rk4::solve;

/// ==============================================================================================
/// ===================================== Experiment Runner ======================================
/// ==============================================================================================

/// Run `num_epochs` trajectories back-to-back, persisting each epoch to disk.
///     Inputs:
///         - `interaction_matrix`: V (d×d)
///         - `growth_vector`: optional g (defaults to 0 inside solver)
///         - `epoch_len`: steps per epoch (each produces `epoch_len+1` snapshots)
///         - `num_epochs`: number of epochs to execute (epoch_1, ..., epoch_{num_epochs})
///         - `output_path`: root output directory
pub fn run(
    interaction_matrix: &Array2<f64>,     // V
    growth_vector: Option<&Array1<f64>>,  // g override
    dt: f64,                              // step size
    epoch_len: usize,                     // steps per epoch
    num_epochs: usize,                    // number of epochs
    output_path: &Path,                   // root output dir
) -> Result<()> {
    let d = interaction_matrix.nrows();
    debug_assert_eq!(interaction_matrix.ncols(), d, "interaction_matrix must be square");
    if let Some(g) = growth_vector {
        debug_assert_eq!(g.len(), d, "growth_vector length must match V");
    }

    // Initial condition: well-mixed uniform simplex (ν_i = 1/d).
    let mode = Mode::Frequency { cutoff: 0.0 };
    let mut gs = create_well_mixed_gs(mode, d, None);

    // Sequential epochs: carry final state from epoch k into epoch k+1.
    for epoch in 1..=num_epochs {
        gs = solve(
            epoch,               // current epoch
            gs,                  // initial state for this epoch
            interaction_matrix,  // V
            growth_vector,       // g
            Noise::none(),       // deterministic run
            dt,                  // step size
            epoch_len,           // steps
            &output_path,  // output target for this epoch
        )?;
    }

    Ok(())
}
