use glam::Vec3;

/// 3D Rotary Position Embedding (3D RoPE) for spatial coordinates (x, y, z).
#[derive(Debug, Clone)]
pub struct RoPE3D {
    pub dim: usize,
    pub base: f32,
}

impl RoPE3D {
    pub fn new(dim: usize, base: f32) -> Self {
        assert_eq!(dim % 6, 0, "Dimension must be divisible by 6 for 3D RoPE");
        Self { dim, base }
    }

    /// Computes rotation angles for 3D coordinate (x, y, z).
    pub fn compute_cos_sin(&self, coord: Vec3) -> (Vec<f32>, Vec<f32>) {
        let dim_per_axis = self.dim / 3;
        let mut cos_all = Vec::with_capacity(self.dim);
        let mut sin_all = Vec::with_capacity(self.dim);

        for (_axis_idx, &val) in [coord.x, coord.y, coord.z].iter().enumerate() {
            for i in 0..(dim_per_axis / 2) {
                let freq = 1.0 / self.base.powf((2 * i) as f32 / dim_per_axis as f32);
                let angle = val * freq;
                cos_all.push(angle.cos());
                cos_all.push(angle.cos());
                sin_all.push(angle.sin());
                sin_all.push(angle.sin());
            }
        }

        (cos_all, sin_all)
    }

    /// Applies 3D rotary embedding to input vector q.
    pub fn apply(&self, q: &mut [f32], coord: Vec3) {
        let (cos, sin) = self.compute_cos_sin(coord);
        for i in (0..self.dim).step_by(2) {
            let x0 = q[i];
            let x1 = q[i + 1];
            q[i] = x0 * cos[i] - x1 * sin[i];
            q[i + 1] = x0 * sin[i + 1] + x1 * cos[i + 1];
        }
    }
}

/// Adaptive Layer Normalization with Zero Initialization (AdaLN-Zero) modulation.
/// Computes: y = (1 + scale) * norm(x) + shift, gate * mlp(norm(y))
#[derive(Debug, Clone)]
pub struct AdaLNZeroParams {
    pub shift_msa: Vec<f32>,
    pub scale_msa: Vec<f32>,
    pub gate_msa: Vec<f32>,
    pub shift_mlp: Vec<f32>,
    pub scale_mlp: Vec<f32>,
    pub gate_mlp: Vec<f32>,
}

impl AdaLNZeroParams {
    pub fn from_embedding(embedding: &[f32], hidden_dim: usize) -> Self {
        let mut params = Self {
            shift_msa: vec![0.0; hidden_dim],
            scale_msa: vec![0.0; hidden_dim],
            gate_msa: vec![0.0; hidden_dim],
            shift_mlp: vec![0.0; hidden_dim],
            scale_mlp: vec![0.0; hidden_dim],
            gate_mlp: vec![0.0; hidden_dim],
        };

        // Linear projection simulation
        for i in 0..hidden_dim {
            let e = if !embedding.is_empty() {
                embedding[i % embedding.len()]
            } else {
                0.0
            };
            params.shift_msa[i] = e * 0.1;
            params.scale_msa[i] = e * 0.1;
            params.gate_msa[i] = (e * 0.05).tanh();
            params.shift_mlp[i] = e * 0.1;
            params.scale_mlp[i] = e * 0.1;
            params.gate_mlp[i] = (e * 0.05).tanh();
        }

        params
    }

    pub fn modulate(&self, x: &[f32], shift: &[f32], scale: &[f32]) -> Vec<f32> {
        x.iter()
            .zip(shift.iter())
            .zip(scale.iter())
            .map(|((&val, &s), &sc)| val * (1.0 + sc) + s)
            .collect()
    }
}

/// Scaled Dot-Product Multi-Head Attention.
pub fn scaled_dot_product_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    d_k: usize,
) -> Vec<f32> {
    let scale = 1.0 / (d_k as f32).sqrt();
    let mut out = vec![0.0; seq_len * d_k];

    for i in 0..seq_len {
        let q_row = &q[i * d_k..(i + 1) * d_k];
        let mut scores = Vec::with_capacity(seq_len);

        for j in 0..seq_len {
            let k_row = &k[j * d_k..(j + 1) * d_k];
            let mut dot = 0.0f32;
            for d in 0..d_k {
                dot += q_row[d] * k_row[d];
            }
            scores.push(dot * scale);
        }

        // Softmax
        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
        let sum_exp: f32 = exp_scores.iter().sum();
        let attn_weights: Vec<f32> = exp_scores.iter().map(|&e| e / sum_exp).collect();

        // Weighted sum of V
        for d in 0..d_k {
            let mut val = 0.0f32;
            for j in 0..seq_len {
                val += attn_weights[j] * v[j * d_k + d];
            }
            out[i * d_k + d] = val;
        }
    }

    out
}

/// Top-k Mixture of Experts (MoE) layer matching Hunyuan3D-DiT.
#[derive(Debug, Clone)]
pub struct MoELayer {
    pub num_experts: usize,
    pub top_k: usize,
    pub hidden_dim: usize,
}

impl MoELayer {
    pub fn new(num_experts: usize, top_k: usize, hidden_dim: usize) -> Self {
        Self {
            num_experts,
            top_k,
            hidden_dim,
        }
    }

    /// Computes top-k gating weights and routes tokens through experts.
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0; x.len()];
        let tokens = x.len() / self.hidden_dim;

        for t in 0..tokens {
            let token = &x[t * self.hidden_dim..(t + 1) * self.hidden_dim];

            // Gate logits (simulated dot product with expert gate embeddings)
            let mut logits = Vec::with_capacity(self.num_experts);
            for e in 0..self.num_experts {
                let mut sum = 0.0f32;
                for (i, &val) in token.iter().enumerate() {
                    let weight = (((e + 1) * 31 + i * 17) % 100) as f32 / 100.0 - 0.5;
                    sum += val * weight;
                }
                logits.push((e, sum));
            }

            // Top-k selection
            logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let top = &logits[0..self.top_k];

            // Softmax over top-k
            let max_val = top.iter().map(|item| item.1).fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = top.iter().map(|item| (item.1 - max_val).exp()).collect();
            let sum_exp: f32 = exps.iter().sum();

            // Route to selected experts
            for (k_idx, &(expert_id, _)) in top.iter().enumerate() {
                let weight = exps[k_idx] / sum_exp;
                let expert_scale = 1.0 + (expert_id as f32) * 0.02;

                for d in 0..self.hidden_dim {
                    let expert_out = (token[d] * expert_scale).tanh();
                    out[t * self.hidden_dim + d] += weight * expert_out;
                }
            }
        }

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_rope_3d() {
        let rope = RoPE3D::new(12, 10000.0);
        let mut q = vec![1.0; 12];
        let coord = Vec3::new(1.0, 2.0, 3.0);

        rope.apply(&mut q, coord);
        // Vector norm should be preserved under orthogonal rotation
        let norm_before = (12.0f32).sqrt();
        let norm_after: f32 = q.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert_relative_eq!(norm_before, norm_after, epsilon = 1e-4);
    }

    #[test]
    fn test_scaled_dot_product_attention() {
        let seq_len = 2;
        let d_k = 4;
        let q = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let k = q.clone();
        let v = vec![0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0];

        let out = scaled_dot_product_attention(&q, &k, &v, seq_len, d_k);
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn test_moe_layer() {
        let moe = MoELayer::new(8, 2, 16);
        let x = vec![0.5; 32]; // 2 tokens of 16-dim
        let out = moe.forward(&x);
        assert_eq!(out.len(), 32);
        assert!(out.iter().any(|&v| v != 0.0));
    }
}
