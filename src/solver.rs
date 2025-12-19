// solver.rs
//! =============================================================================================
//! Replicator solver: RK4 replicator dynamics on the simplex + optional stochastic noise
//! =============================================================================================
//!
//! This module implements replicator-type dynamics directly in **frequency space**:
//!     - State ν lies on the probability simplex (nonnegative, sums to 1).
//!     - Deterministic integration uses explicit RK4 on the replicator RHS.
//!     - Optional stochasticity is applied *after* the deterministic step.
//!     - After every operation that can drift, we call `Simplex::sanitize()`.
//!
//! OUTPUT
//! ------
//! Frequencies are written into `TaxonTable<f64>` and saved as JSON into:
//!     {dir_output}/table/{t}.json
//!
//! PROJECT ASSUMPTIONS
//! -------------------
//! - This project is standalone and does NOT depend on physics-in-parallel.
//! - This module depends only on the local modules:
//!     - `simplex.rs`  (Simplex)
//!     - `noise.rs`    (Noise, NoiseContext, apply_noise_inplace)
//!     - `taxon_table.rs` (TaxonTable JSON I/O)
//!
//! =============================================================================================
#![allow(dead_code)]

use std::collections::VecDeque;
use std::fs::create_dir_all;
use std::path::PathBuf;

use ndarray::{Array1, Array2};

use rand::rngs::SmallRng;
use rand::SeedableRng;

use crate::noise::{apply_noise_inplace, Noise, NoiseContext};
use crate::simplex::Simplex;
use crate::taxon_table::TaxonTable;


/// ==============================================================================================
/// =========================== Deterministic core: replicator RHS ================================
/// ==============================================================================================

/// Scratch buffers for RK4 (to avoid repeated allocations).
struct Rk4Scratch {
    k1: Array1<f64>,
    k2: Array1<f64>,
    k3: Array1<f64>,
    k4: Array1<f64>,
    tmp: Array1<f64>,
    w: Array1<f64>,     // holds V·ν
    drift: Array1<f64>, // holds s + w - Υ
}

impl Rk4Scratch {
    #[inline]
    fn new(d: usize) -> Self {
        Self {
            k1: Array1::zeros(d),
            k2: Array1::zeros(d),
            k3: Array1::zeros(d),
            k4: Array1::zeros(d),
            tmp: Array1::zeros(d),
            w: Array1::zeros(d),
            drift: Array1::zeros(d),
        }
    }
}

/// Compute the replicator RHS in-place:
///
///     out = rhs(ν) = ν ⊙ ( s + Vν - Υ ), where Υ = Σ_i ν_i (s_i + (Vν)_i)
///
/// This ODE conserves total mass analytically; numerical drift is corrected by `Simplex::sanitize()`.
#[inline]
fn rhs_inplace(
    nu: &Array1<f64>,
    s: &Array1<f64>,
    v: &Array2<f64>,
    w: &mut Array1<f64>,
    drift: &mut Array1<f64>,
    out: &mut Array1<f64>,
) {
    let d = nu.len();

    debug_assert_eq!(s.len(), d, "rhs_inplace: s length mismatch");
    debug_assert_eq!(v.nrows(), d, "rhs_inplace: V rows mismatch");
    debug_assert_eq!(v.ncols(), d, "rhs_inplace: V cols mismatch");

    // ------------------------------------------------------------------------------------------
    // (1) w = V · nu
    // ------------------------------------------------------------------------------------------
    for i in 0..d {
        let mut acc = 0.0;
        for j in 0..d {
            acc += v[(i, j)] * nu[j];
        }
        w[i] = acc;
    }

    // ------------------------------------------------------------------------------------------
    // (2) Υ = Σ_i ν_i (s_i + w_i)
    // ------------------------------------------------------------------------------------------
    let mut upsilon = 0.0;
    for i in 0..d {
        upsilon += nu[i] * (s[i] + w[i]);
    }

    // ------------------------------------------------------------------------------------------
    // (3) drift = s + w - Υ
    // ------------------------------------------------------------------------------------------
    for i in 0..d {
        drift[i] = s[i] + w[i] - upsilon;
    }

    // ------------------------------------------------------------------------------------------
    // (4) out = ν ⊙ drift
    // ------------------------------------------------------------------------------------------
    for i in 0..d {
        out[i] = nu[i] * drift[i];
    }
}

/// One explicit RK4 step (deterministic) writing into `out`.
///
/// Notes:
///     - This routine does NOT enforce the simplex invariant.
///     - It only clamps non-finite/negative outputs to 0.
///     - The caller must call `Simplex::sanitize()` immediately afterward.
#[inline]
fn rk4_step_inplace_raw(
    nu: &Array1<f64>,
    s: &Array1<f64>,
    v: &Array2<f64>,
    dt: f64,
    sc: &mut Rk4Scratch,
    out: &mut Array1<f64>,
) {
    let d = nu.len();
    debug_assert_eq!(s.len(), d, "rk4_step_inplace_raw: s length mismatch");
    debug_assert_eq!(v.nrows(), d, "rk4_step_inplace_raw: V rows mismatch");
    debug_assert_eq!(v.ncols(), d, "rk4_step_inplace_raw: V cols mismatch");

    let half_dt = 0.5 * dt;
    let dt_over_6 = dt / 6.0;

    // k1 = rhs(nu)
    rhs_inplace(nu, s, v, &mut sc.w, &mut sc.drift, &mut sc.k1);

    // tmp = nu + 0.5*dt*k1
    for i in 0..d {
        sc.tmp[i] = nu[i] + half_dt * sc.k1[i];
    }
    // k2 = rhs(tmp)
    rhs_inplace(&sc.tmp, s, v, &mut sc.w, &mut sc.drift, &mut sc.k2);

    // tmp = nu + 0.5*dt*k2
    for i in 0..d {
        sc.tmp[i] = nu[i] + half_dt * sc.k2[i];
    }
    // k3 = rhs(tmp)
    rhs_inplace(&sc.tmp, s, v, &mut sc.w, &mut sc.drift, &mut sc.k3);

    // tmp = nu + dt*k3
    for i in 0..d {
        sc.tmp[i] = nu[i] + dt * sc.k3[i];
    }
    // k4 = rhs(tmp)
    rhs_inplace(&sc.tmp, s, v, &mut sc.w, &mut sc.drift, &mut sc.k4);

    // out = nu + dt/6*(k1 + 2k2 + 2k3 + k4)
    for i in 0..d {
        let incr = dt_over_6 * (sc.k1[i] + 2.0 * sc.k2[i] + 2.0 * sc.k3[i] + sc.k4[i]);
        let mut val = nu[i] + incr;
        if !val.is_finite() || val <= 0.0 {
            val = 0.0;
        }
        out[i] = val;
    }
}



/// ==============================================================================================
/// ============================== Solver as a stateful struct ===================================
/// ==============================================================================================

/// Keep a short rolling buffer of recent states (for debugging/inspection),
/// instead of storing the full trajectory.
const RING_KEEP: usize = 32; // how many snapshots to keep
const RING_STRIDE: usize = 10_000; // sample every N steps

/// Stateful solver that owns all buffers, RNG, and output paths.
///
/// Minimum usage:
///     - Construct with `new(...)`
///     - Call `run()` to advance to `max_steps`
///
/// You can also drive it manually via repeated calls to `step(step_idx)`.
pub struct Solver {
    // IO
    dir_output: PathBuf,

    // Model (owned)
    d: usize,
    v: Array2<f64>,
    s: Array1<f64>,

    // Run configuration
    dt: f64,
    max_steps: usize,
    noise: Noise,
    save_interval: usize,

    // State
    nu: Simplex,
    nu_next: Simplex,

    // Output table
    table: TaxonTable<f64>,
    last_saved_step: usize,

    // Workspaces (reused each step)
    rk4: Rk4Scratch,
    noise_ctx: NoiseContext,
    rng: SmallRng,

    // Diagnostics
    recent_states: VecDeque<Vec<f64>>,
    recent_l2: VecDeque<f64>,
}

impl Solver {
    /// Construct a new solver and perform the initial save at t=0.
    ///
    /// Inputs:
    ///     - `dir_output`: output root directory; this solver writes into `{dir_output}/table/`
    ///     - `v`: interaction matrix V (d×d)
    ///     - `s`: selection/payoff vector s (length d); if `None`, defaults to zeros
    ///     - `nu_init`: initial simplex point; if `None`, defaults to uniform
    ///     - `dt`: timestep
    ///     - `max_steps`: number of integration steps
    ///     - `noise`: noise configuration
    ///     - `save_interval`: if > 0, save every `save_interval` steps
    ///     - `cutoff`: absorbing boundary stored in `Simplex`
    pub fn new(
        dir_output: &PathBuf,
        v: &Array2<f64>,
        s: Option<&Array1<f64>>,
        nu_init: Option<&[f64]>,
        dt: f64,
        max_steps: usize,
        noise: Noise,
        save_interval: usize,
        cutoff: f64,
    ) -> Self {
        // --------------------------------------------------------------------------------------
        // (1) Validate + materialize model arrays
        // --------------------------------------------------------------------------------------
        assert_eq!(v.nrows(), v.ncols(), "ReplicatorSolver::new: V must be square");
        let d = v.nrows();
        assert!(d > 0, "ReplicatorSolver::new: d must be > 0");

        let v_owned = v.to_owned();
        let s_owned: Array1<f64> = s.map(|x| x.to_owned()).unwrap_or_else(|| Array1::zeros(d));
        assert_eq!(s_owned.len(), d, "ReplicatorSolver::new: s length mismatch");

        // --------------------------------------------------------------------------------------
        // (2) Initialize ν(0) as a Simplex (stores cutoff internally)
        // --------------------------------------------------------------------------------------
        let nu: Simplex = if let Some(nu0) = nu_init {
            assert_eq!(nu0.len(), d, "ReplicatorSolver::new: nu_init length mismatch");
            Simplex::from_slice(nu0, cutoff)
        } else {
            Simplex::uniform(d, cutoff)
        };
        let nu_next: Simplex = Simplex::uniform(d, cutoff);

        // --------------------------------------------------------------------------------------
        // (3) Output dirs
        // --------------------------------------------------------------------------------------
        let dir_output = dir_output.clone();
        let _ = create_dir_all(&dir_output);

        // --------------------------------------------------------------------------------------
        // (4) Output table init + initial save (0.json)
        // --------------------------------------------------------------------------------------
        let mut table = TaxonTable::<f64>::new(d);
        nu.write_into_table(&mut table);
        let _ = table.save_to_json(&dir_output.join("0.json"));

        // --------------------------------------------------------------------------------------
        // (5) Construct solver
        // --------------------------------------------------------------------------------------
        Self {
            dir_output,

            d,
            v: v_owned,
            s: s_owned,

            dt,
            max_steps,
            noise,
            save_interval,

            nu,
            nu_next,

            table,
            last_saved_step: 0,

            rk4: Rk4Scratch::new(d),
            noise_ctx: NoiseContext::new(d),
            rng: SmallRng::try_from_os_rng().expect("Failed to initialize RNG"),

            recent_states: VecDeque::with_capacity(RING_KEEP),
            recent_l2: VecDeque::with_capacity(RING_KEEP),
        }
    }

    /// Access the output directory.
    #[inline]
    pub fn dir_output(&self) -> &PathBuf {
        &self.dir_output
    }

    /// Access the current simplex state ν.
    #[inline]
    pub fn state(&self) -> &Simplex {
        &self.nu
    }

    /// Access the internal output table (contains last written state).
    #[inline]
    pub fn table(&self) -> &TaxonTable<f64> {
        &self.table
    }

    /// Save the current ν into the `TaxonTable` and write `{step}.json`.
    #[inline]
    fn save_step(&mut self, step: usize) {
        self.nu.write_into_table(&mut self.table);
        let _ = self
            .table
            .save_to_json(&self.dir_output.join(format!("{step}.json")));
        self.last_saved_step = step;
    }

    /// Perform one time step. Returns the L2 difference ||ν_next - ν|| computed before the swap.
    pub fn step(&mut self, step: usize) -> f64 {
        debug_assert!(step >= 1 && step <= self.max_steps);

        // --------------------------------------------------------------------------------------
        // (a) Deterministic RK4: ν -> ν_next (raw)
        // --------------------------------------------------------------------------------------
        rk4_step_inplace_raw(
            self.nu.as_array(),
            &self.s,
            &self.v,
            self.dt,
            &mut self.rk4,
            self.nu_next.as_array_mut(),
        );

        // Restore simplex constraints after deterministic integration.
        self.nu_next.sanitize();

        // --------------------------------------------------------------------------------------
        // (b) Optional stochastic step (always ends with sanitize inside)
        // --------------------------------------------------------------------------------------
        apply_noise_inplace(
            &mut self.nu_next,
            self.noise,
            self.dt,
            &mut self.noise_ctx,
            &mut self.rng,
        );

        // --------------------------------------------------------------------------------------
        // (c) Diagnostics: L2 difference ||ν_next - ν||
        // --------------------------------------------------------------------------------------
        let mut acc = 0.0f64;
        {
            let a = self.nu.as_array();
            let b = self.nu_next.as_array();
            for i in 0..self.d {
                let dd = b[i] - a[i];
                acc += dd * dd;
            }
        }
        let diff = acc.sqrt();

        // --------------------------------------------------------------------------------------
        // (d) Optional ring history
        // --------------------------------------------------------------------------------------
        if RING_STRIDE > 0 && step % RING_STRIDE == 0 {
            if self.recent_states.len() == RING_KEEP {
                self.recent_states.pop_front();
            }
            if self.recent_l2.len() == RING_KEEP {
                self.recent_l2.pop_front();
            }
            self.recent_states.push_back(self.nu_next.as_array().to_vec());
            self.recent_l2.push_back(diff);
        }

        // --------------------------------------------------------------------------------------
        // (e) Advance: swap ν and ν_next without allocation
        // --------------------------------------------------------------------------------------
        std::mem::swap(&mut self.nu, &mut self.nu_next);

        // --------------------------------------------------------------------------------------
        // (f) Periodic save
        // --------------------------------------------------------------------------------------
        if self.save_interval > 0 && step % self.save_interval == 0 {
            self.save_step(step);
        }

        diff
    }

    /// Run the solver to completion and return the final `TaxonTable<f64>`.
    pub fn run(mut self) -> TaxonTable<f64> {
        for step in 1..=self.max_steps {
            let _ = self.step(step);
        }

        // Final save if not already saved at `max_steps`
        if self.last_saved_step != self.max_steps {
            self.save_step(self.max_steps);
        }

        self.table
    }
}
