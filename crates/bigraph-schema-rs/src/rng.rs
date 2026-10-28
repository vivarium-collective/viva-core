//! Bit-for-bit reimplementation of numpy's legacy `RandomState`.
//!
//! numpy's legacy `RandomState` is a standard **MT19937** bit generator plus
//! numpy's **own** per-distribution algorithms (the randomkit / `distributions.c`
//! lineage). Off-the-shelf MT crates reproduce the *bit-generator* layer but
//! **not** the distribution layer (`rand`/`rand_distr` and the `Generator`/PCG64
//! path all differ). v2ecoli's stochastic processes must stay bit-identical to
//! their Python twins, so this module reimplements numpy's algorithms exactly:
//!
//! * seeding   — `mt19937_seed` (a.k.a. `init_genrand`) for scalar int seeds,
//!   which is the only path v2ecoli exercises (`RandomState(seed=<small int>)`).
//! * `random_sample` / `rand` — `genrand_res53` (two 32-bit words -> 53-bit f64).
//! * `stochastic_round`        — v2ecoli `library/random.py::stochasticRound`
//!   (one batched `rand` draw, `u < frac(v)` -> ceil else floor).
//! * `poisson`                 — numpy's two-regime algorithm
//!   (`rk_poisson_mult` for lam<10, `rk_poisson_ptrs` for lam>=10, 0 for lam==0)
//!   with the Lanczos `loggam` matching numpy's constants.
//!
//! Validated against a numpy oracle in `tests/rng_parity.rs`
//! (`tests/oracle/gen_rng_oracle.py`), f64 compared by exact bit pattern.

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UPPER_MASK: u32 = 0x8000_0000;
const LOWER_MASK: u32 = 0x7fff_ffff;

/// numpy legacy `RandomState` (MT19937 core + numpy's distribution algorithms).
#[derive(Clone)]
pub struct RandomState {
    key: [u32; N],
    pos: usize,
}

impl RandomState {
    /// Seed exactly as numpy does for a scalar int seed: `mt19937_seed`
    /// (`init_genrand`). `key[0] = seed`, then
    /// `key[i] = 1812433253 * (key[i-1] ^ (key[i-1] >> 30)) + i`, all mod 2^32.
    ///
    /// v2ecoli always seeds with a small int (`_seedFromName` -> scalar), so
    /// this is the only seeding path needed; `init_by_array` (array / >=2^32
    /// seeds) is deliberately not implemented.
    pub fn new(seed: u32) -> Self {
        let mut key = [0u32; N];
        let mut s = seed;
        for (pos, slot) in key.iter_mut().enumerate() {
            *slot = s;
            s = 1812433253u32
                .wrapping_mul(s ^ (s >> 30))
                .wrapping_add(pos as u32 + 1);
        }
        RandomState { key, pos: N }
    }

    /// First three words of the freshly-seeded MT state — a direct seeding check
    /// against numpy's `get_state()[1][0:3]`. Only meaningful before any draw.
    pub fn key_prefix(&self) -> [u32; 3] {
        [self.key[0], self.key[1], self.key[2]]
    }

    /// Regenerate the 624-word MT block (standard MT19937 twist).
    fn generate(&mut self) {
        for i in 0..(N - M) {
            let y = (self.key[i] & UPPER_MASK) | (self.key[i + 1] & LOWER_MASK);
            self.key[i] = self.key[i + M] ^ (y >> 1) ^ (if y & 1 != 0 { MATRIX_A } else { 0 });
        }
        for i in (N - M)..(N - 1) {
            let y = (self.key[i] & UPPER_MASK) | (self.key[i + 1] & LOWER_MASK);
            self.key[i] = self.key[i + M - N] ^ (y >> 1) ^ (if y & 1 != 0 { MATRIX_A } else { 0 });
        }
        let y = (self.key[N - 1] & UPPER_MASK) | (self.key[0] & LOWER_MASK);
        self.key[N - 1] = self.key[M - 1] ^ (y >> 1) ^ (if y & 1 != 0 { MATRIX_A } else { 0 });
        self.pos = 0;
    }

    /// One tempered 32-bit MT word (`mt19937_next`).
    #[inline]
    fn next_u32(&mut self) -> u32 {
        if self.pos == N {
            self.generate();
        }
        let mut y = self.key[self.pos];
        self.pos += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    /// `genrand_res53`: two words -> a 53-bit double in [0, 1). This is numpy's
    /// legacy `rk_double` / `next_double`, identical to `rand`/`random_sample`.
    #[inline]
    pub fn next_double(&mut self) -> f64 {
        let a = (self.next_u32() >> 5) as f64;
        let b = (self.next_u32() >> 6) as f64;
        (a * 67108864.0 + b) / 9007199254740992.0
    }

    /// `random_sample(n)` / `rand(n)` — `n` consecutive res53 draws, in order.
    pub fn random_sample(&mut self, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.next_double()).collect()
    }

    /// v2ecoli `stochasticRound` (1-D C-order path): one batched `rand(size)`
    /// draw, then `ceil` where `u < frac(v)` else `floor`. Returns f64 (as numpy
    /// `ceil`/`floor` do); the values are integral.
    pub fn stochastic_round(&mut self, values: &[f64]) -> Vec<f64> {
        let draws = self.random_sample(values.len());
        values
            .iter()
            .zip(draws)
            .map(|(&v, u)| {
                // numpy `v % 1` == v - floor(v) for all finite v.
                let frac = v - v.floor();
                if u < frac {
                    v.ceil()
                } else {
                    v.floor()
                }
            })
            .collect()
    }

    /// numpy `poisson(lam)` scalar: regime split at lam == 10, lam == 0 -> 0
    /// (draws nothing), matching `rk_poisson`.
    pub fn poisson(&mut self, lam: f64) -> i64 {
        if lam >= 10.0 {
            self.poisson_ptrs(lam)
        } else if lam == 0.0 {
            0
        } else {
            self.poisson_mult(lam)
        }
    }

    /// `poisson(lam, size=n)` — `n` scalar draws on the shared stream, in order.
    pub fn poisson_n(&mut self, lam: f64, n: usize) -> Vec<i64> {
        (0..n).map(|_| self.poisson(lam)).collect()
    }

    /// Vector-`lam` `poisson` — one draw per element, C order, shared stream
    /// (numpy consumes left-to-right; lam==0 elements draw nothing).
    pub fn poisson_each(&mut self, lams: &[f64]) -> Vec<i64> {
        lams.iter().map(|&lam| self.poisson(lam)).collect()
    }

    /// `rk_poisson_mult` — Knuth multiplication method (lam < 10).
    fn poisson_mult(&mut self, lam: f64) -> i64 {
        let enlam = (-lam).exp();
        let mut x: i64 = 0;
        let mut prod = 1.0f64;
        loop {
            let u = self.next_double();
            prod *= u;
            if prod > enlam {
                x += 1;
            } else {
                return x;
            }
        }
    }

    /// `rk_poisson_ptrs` — transformed-rejection method (lam >= 10).
    fn poisson_ptrs(&mut self, lam: f64) -> i64 {
        let slam = lam.sqrt();
        let loglam = lam.ln();
        let b = 0.931 + 2.53 * slam;
        let a = -0.059 + 0.02483 * b;
        let invalpha = 1.1239 + 1.1328 / (b - 3.4);
        let vr = 0.9277 - 3.6224 / (b - 2.0);
        loop {
            let u = self.next_double() - 0.5;
            let v = self.next_double();
            let us = 0.5 - u.abs();
            let k = ((2.0 * a / us + b) * u + lam + 0.43).floor() as i64;
            if us >= 0.07 && v <= vr {
                return k;
            }
            if k < 0 || (us < 0.013 && v > us) {
                continue;
            }
            if (v.ln() + invalpha.ln() - (a / (us * us) + b).ln())
                <= (-lam + (k as f64) * loglam - loggam((k + 1) as f64))
            {
                return k;
            }
        }
    }
}

/// numpy `rk_loggam` — log Gamma via the Lanczos series with numpy's exact
/// coefficients and the `x <= 7` recurrence shift.
fn loggam(x: f64) -> f64 {
    // numpy's distributions.c coefficients, verbatim; one literal carries a
    // redundant trailing digit (same f64 bits) — keep it to match the source.
    #[allow(clippy::excessive_precision)]
    const A: [f64; 10] = [
        8.333333333333333e-02,
        -2.777777777777778e-03,
        7.936507936507937e-04,
        -5.952380952380952e-04,
        8.417508417508418e-04,
        -1.917526917526918e-03,
        6.410256410256410e-03,
        -2.955065359477124e-02,
        1.796443723688307e-01,
        -1.39243221690590e+00,
    ];
    if x == 1.0 || x == 2.0 {
        return 0.0;
    }
    let mut x0 = x;
    let mut n = 0i64;
    if x <= 7.0 {
        n = (7.0 - x) as i64;
        x0 = x + n as f64;
    }
    let x2 = 1.0 / (x0 * x0);
    let xp = 2.0 * std::f64::consts::PI;
    let mut gl0 = A[9];
    for k in (0..9).rev() {
        gl0 *= x2;
        gl0 += A[k];
    }
    let mut gl = gl0 / x0 + 0.5 * xp.ln() + (x0 - 0.5) * x0.ln() - x0;
    if x <= 7.0 {
        for _ in 1..=n {
            gl -= (x0 - 1.0).ln();
            x0 -= 1.0;
        }
    }
    gl
}
