use crate::error::Result;
use serde::{Deserialize, Serialize};

/// Type of ODE solver for Flow Matching sampling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ODESolverType {
    Euler,
    Midpoint,
    Heun,
    Rk4,
}

/// Noise and time schedule configuration for Rectified Flow.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowSchedule {
    pub num_steps: usize,
    pub shift: f32, // Resolution shift factor (default 3.0 matching Hunyuan3D-2.1)
    pub guidance_scale: f32, // Classifier-free guidance (default 5.0)
}

impl Default for FlowSchedule {
    fn default() -> Self {
        Self {
            num_steps: 5,
            shift: 3.0,
            guidance_scale: 5.0,
        }
    }
}

impl FlowSchedule {
    /// Computes shifted sigma schedule values for diffusion / flow matching.
    /// σ_t = (shift * t) / (1 + (shift - 1) * t)
    pub fn compute_sigmas(&self) -> Vec<f32> {
        let n = self.num_steps;
        let mut sigmas = Vec::with_capacity(n + 1);

        for i in 0..=n {
            let t = 1.0 - (i as f32) / (n as f32); // From 1.0 down to 0.0
            let shifted = (self.shift * t) / (1.0 + (self.shift - 1.0) * t);
            sigmas.push(shifted);
        }

        sigmas
    }
}

/// Rectified Flow ODE Solver for Hunyuan3D.
pub struct RectifiedFlowSolver {
    pub solver_type: ODESolverType,
    pub schedule: FlowSchedule,
}

impl RectifiedFlowSolver {
    pub fn new(solver_type: ODESolverType, schedule: FlowSchedule) -> Self {
        Self {
            solver_type,
            schedule,
        }
    }

    /// Solves the Flow Matching ODE from initial noise x_1 to clean sample x_0.
    pub fn solve<F>(&self, initial_sample: &[f32], mut velocity_fn: F) -> Result<Vec<f32>>
    where
        F: FnMut(&[f32], f32, &mut [f32]) -> Result<()>,
    {
        let sigmas = self.schedule.compute_sigmas();
        let dim = initial_sample.len();
        let mut x = initial_sample.to_vec();

        let mut v1 = vec![0.0; dim];
        let mut v2 = vec![0.0; dim];
        let mut v3 = vec![0.0; dim];
        let mut v4 = vec![0.0; dim];
        let mut temp_x = vec![0.0; dim];

        for step in 0..self.schedule.num_steps {
            let s_curr = sigmas[step];
            let s_next = sigmas[step + 1];
            let dt = s_next - s_curr; // negative value

            match self.solver_type {
                ODESolverType::Euler => {
                    velocity_fn(&x, s_curr, &mut v1)?;
                    for i in 0..dim {
                        x[i] += dt * v1[i];
                    }
                }
                ODESolverType::Midpoint => {
                    velocity_fn(&x, s_curr, &mut v1)?;
                    let half_dt = dt * 0.5;
                    for i in 0..dim {
                        temp_x[i] = x[i] + half_dt * v1[i];
                    }
                    velocity_fn(&temp_x, s_curr + half_dt, &mut v2)?;
                    for i in 0..dim {
                        x[i] += dt * v2[i];
                    }
                }
                ODESolverType::Heun => {
                    velocity_fn(&x, s_curr, &mut v1)?;
                    for i in 0..dim {
                        temp_x[i] = x[i] + dt * v1[i];
                    }
                    velocity_fn(&temp_x, s_next, &mut v2)?;
                    for i in 0..dim {
                        x[i] += dt * 0.5 * (v1[i] + v2[i]);
                    }
                }
                ODESolverType::Rk4 => {
                    // k1
                    velocity_fn(&x, s_curr, &mut v1)?;

                    // k2
                    let half_dt = dt * 0.5;
                    for i in 0..dim {
                        temp_x[i] = x[i] + half_dt * v1[i];
                    }
                    velocity_fn(&temp_x, s_curr + half_dt, &mut v2)?;

                    // k3
                    for i in 0..dim {
                        temp_x[i] = x[i] + half_dt * v2[i];
                    }
                    velocity_fn(&temp_x, s_curr + half_dt, &mut v3)?;

                    // k4
                    for i in 0..dim {
                        temp_x[i] = x[i] + dt * v3[i];
                    }
                    velocity_fn(&temp_x, s_next, &mut v4)?;

                    for i in 0..dim {
                        x[i] += (dt / 6.0) * (v1[i] + 2.0 * v2[i] + 2.0 * v3[i] + v4[i]);
                    }
                }
            }
        }

        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_flow_schedule() {
        let sched = FlowSchedule::default();
        let sigmas = sched.compute_sigmas();
        assert_eq!(sigmas.len(), 6);
        assert_relative_eq!(sigmas[0], 1.0, epsilon = 1e-4);
        assert_relative_eq!(sigmas[5], 0.0, epsilon = 1e-4);
    }

    #[test]
    fn test_ode_linear_integration() {
        // v(x, t) = -x -> exact solution: x(t) = x0 * exp(t - 1)
        let sched = FlowSchedule {
            num_steps: 10,
            shift: 1.0,
            guidance_scale: 1.0,
        };
        let solver = RectifiedFlowSolver::new(ODESolverType::Heun, sched);

        let initial = vec![1.0f32, 2.0f32];
        let out = solver
            .solve(&initial, |x, _t, v| {
                for i in 0..x.len() {
                    v[i] = -x[i];
                }
                Ok(())
            })
            .unwrap();

        assert_eq!(out.len(), 2);
        assert!(out[0] > 1.0);
        assert!((out[0] - std::f32::consts::E).abs() < 0.2);
    }
}
