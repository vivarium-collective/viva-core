//! Bit-for-bit parity of `bigraph_schema_rs::rng::RandomState` vs numpy's legacy
//! `RandomState`. Expected values are committed literals produced by
//! `tests/oracle/gen_rng_oracle.py` (numpy 2.4.6), re-run as:
//!
//! ```text
//! /Users/eranagmon/code/v2ecoli/.venv/bin/python \
//!     crates/bigraph-schema-rs/tests/oracle/gen_rng_oracle.py
//! ```
//!
//! Floats are compared by exact IEEE-754 bit pattern (`f64::to_bits`), never an
//! epsilon — the RNG stream must be reproduced exactly. A wrong seeding, a
//! wrong res53, a dropped/extra draw, a wrong poisson regime boundary, or a
//! wrong `loggam` all make at least one of these assertions fail.

use bigraph_schema_rs::rng::RandomState;

/// One oracle case per seed, transcribed verbatim from gen_rng_oracle.py output.
struct Case {
    seed: u32,
    key_prefix: [u32; 3],
    random_sample_bits: [u64; 8],
    // stochastic_round on FRAC_VEC below (integral results).
    stochastic_round: [i64; 8],
    poisson_0: [i64; 6],
    poisson_2_5: [i64; 6],
    poisson_10: [i64; 6],
    poisson_50: [i64; 6],
    poisson_vec: [i64; 4], // lam = [0, 3, 0, 12]
    // interleave: random_sample(3); poisson(7.5, 4); random_sample(2) — one stream.
    interleave_pre_bits: [u64; 3],
    interleave_poisson: [i64; 4],
    interleave_post_bits: [u64; 2],
}

const FRAC_VEC: [f64; 8] = [0.0, 0.25, 0.5, 0.75, 1.0, 2.3, 9.99, 100.5];
const POISSON_VEC_LAM: [f64; 4] = [0.0, 3.0, 0.0, 12.0];

const CASES: &[Case] = &[
    Case {
        seed: 0,
        key_prefix: [0, 1, 1812433255],
        random_sample_bits: [
            0x3fe18fe1565f12a8,
            0x3fe6e2d4cf608733,
            0x3fe349d66b6e894b,
            0x3fe16faedb6395f5,
            0x3fdb1d2904fd0ef6,
            0x3fe4ab2a21899b5b,
            0x3fdc016dca6159d2,
            0x3fec8967883a12f7,
        ],
        stochastic_round: [0, 0, 0, 1, 1, 2, 10, 100],
        poisson_0: [0, 0, 0, 0, 0, 0],
        poisson_2_5: [4, 5, 3, 1, 7, 3],
        poisson_10: [10, 11, 9, 9, 18, 13],
        poisson_50: [51, 52, 48, 49, 69, 57],
        poisson_vec: [0, 5, 0, 11],
        interleave_pre_bits: [0x3fe18fe1565f12a8, 0x3fe6e2d4cf608733, 0x3fe349d66b6e894b],
        interleave_poisson: [11, 7, 7, 9],
        interleave_post_bits: [0x3fdbf854df380288, 0x3fe652fea865a9bf],
    },
    Case {
        seed: 1,
        key_prefix: [1, 1812433254, 3713160357],
        random_sample_bits: [
            0x3fdab07d0ffa3c06,
            0x3fe70ce5f3baf051,
            0x3f1dfb906661c000,
            0x3fd3596ab7fe08f4,
            0x3fc2c8e5a3c7053c,
            0x3fb7a3808cb0ca10,
            0x3fc7d75fe634e1e4,
            0x3fd61daabd5ba3ce,
        ],
        stochastic_round: [0, 0, 1, 1, 1, 3, 10, 101],
        poisson_0: [0, 0, 0, 0, 0, 0],
        poisson_2_5: [2, 1, 1, 2, 2, 1],
        poisson_10: [9, 6, 7, 9, 9, 9],
        poisson_50: [48, 41, 43, 48, 48, 48],
        poisson_vec: [0, 2, 0, 10],
        interleave_pre_bits: [0x3fdab07d0ffa3c06, 0x3fe70ce5f3baf051, 0x3f1dfb906661c000],
        interleave_poisson: [4, 6, 11, 4],
        interleave_post_bits: [0x3feea707f1a5abb8, 0x3fe10fb0a4ca4ac8],
    },
    Case {
        seed: 42,
        key_prefix: [42, 3107752595, 1895908407],
        random_sample_bits: [
            0x3fd7f8771e5f51ec,
            0x3fee6c4068bbd654,
            0x3fe76c7e8f1e6751,
            0x3fe32835d6632cb0,
            0x3fc3f86b37222184,
            0x3fc3f7a0e1997f4c,
            0x3fadbd229d645570,
            0x3febb7b70955b7b5,
        ],
        stochastic_round: [0, 0, 0, 1, 1, 3, 10, 100],
        poisson_0: [0, 0, 0, 0, 0, 0],
        poisson_2_5: [4, 1, 3, 3, 1, 2],
        poisson_10: [12, 6, 11, 14, 7, 8],
        poisson_50: [47, 55, 42, 52, 58, 43],
        poisson_vec: [0, 4, 0, 8],
        interleave_pre_bits: [0x3fd7f8771e5f51ec, 0x3fee6c4068bbd654, 0x3fe76c7e8f1e6751],
        interleave_poisson: [5, 5, 6, 7],
        interleave_post_bits: [0x3fe37101e6b8a28a, 0x3fc5d3bc07353eb4],
    },
    Case {
        seed: 12345,
        key_prefix: [12345, 2003863422, 2690739229],
        random_sample_bits: [
            0x3fedbf6a3f8f84b7,
            0x3fd43f7f410bb044,
            0x3fc78aa6d0a2dad8,
            0x3fca2f07fd391514,
            0x3fe22acdae20d908,
            0x3fe30eb3c3d342ec,
            0x3feedd4d8dd90335,
            0x3fe4e6d3a7b1e86a,
        ],
        stochastic_round: [0, 0, 1, 1, 1, 2, 10, 100],
        poisson_0: [0, 0, 0, 0, 0, 0],
        poisson_2_5: [2, 2, 6, 1, 7, 4],
        poisson_10: [16, 7, 11, 18, 12, 8],
        poisson_50: [63, 43, 51, 55, 46, 53],
        poisson_vec: [0, 3, 0, 13],
        interleave_pre_bits: [0x3fedbf6a3f8f84b7, 0x3fd43f7f410bb044, 0x3fc78aa6d0a2dad8],
        interleave_poisson: [9, 11, 8, 9],
        interleave_post_bits: [0x3fcc068e16a4c010, 0x3fd08eda651bfab8],
    },
];

fn assert_bits(label: &str, seed: u32, got: &[f64], want: &[u64]) {
    assert_eq!(got.len(), want.len(), "{label} len (seed {seed})");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(
            g.to_bits(),
            *w,
            "{label}[{i}] seed {seed}: got {:#018x} ({g}), want {:#018x}",
            g.to_bits(),
            *w,
        );
    }
}

#[test]
fn seeding_matches_numpy_init_genrand() {
    for c in CASES {
        let rs = RandomState::new(c.seed);
        assert_eq!(
            rs.key_prefix(),
            c.key_prefix,
            "MT19937 key[0:3] mismatch for seed {}",
            c.seed
        );
    }
}

#[test]
fn random_sample_matches_numpy_res53() {
    for c in CASES {
        let mut rs = RandomState::new(c.seed);
        let got = rs.random_sample(8);
        assert_bits("random_sample", c.seed, &got, &c.random_sample_bits);
    }
}

#[test]
fn stochastic_round_matches_v2ecoli() {
    for c in CASES {
        let mut rs = RandomState::new(c.seed);
        let got = rs.stochastic_round(&FRAC_VEC);
        let want: Vec<f64> = c.stochastic_round.iter().map(|&x| x as f64).collect();
        assert_bits(
            "stochastic_round",
            c.seed,
            &got,
            &want.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        );
    }
}

#[test]
fn poisson_scalars_match_numpy() {
    for c in CASES {
        for (lam, want) in [
            (0.0, &c.poisson_0[..]),
            (2.5, &c.poisson_2_5[..]),
            (10.0, &c.poisson_10[..]),
            (50.0, &c.poisson_50[..]),
        ] {
            let mut rs = RandomState::new(c.seed);
            let got = rs.poisson_n(lam, want.len());
            assert_eq!(got, want, "poisson(lam={lam}) seed {}", c.seed);
        }
    }
}

#[test]
fn poisson_vector_lam_matches_numpy() {
    for c in CASES {
        let mut rs = RandomState::new(c.seed);
        let got = rs.poisson_each(&POISSON_VEC_LAM);
        assert_eq!(
            got,
            c.poisson_vec.to_vec(),
            "poisson(vec) seed {} (vector lam, C-order shared stream)",
            c.seed
        );
    }
}

#[test]
fn interleaved_stream_advances_across_distributions() {
    for c in CASES {
        let mut rs = RandomState::new(c.seed);
        let pre = rs.random_sample(3);
        assert_bits("interleave_pre", c.seed, &pre, &c.interleave_pre_bits);
        let pois = rs.poisson_n(7.5, 4);
        assert_eq!(
            pois,
            c.interleave_poisson.to_vec(),
            "interleave poisson seed {}",
            c.seed
        );
        let post = rs.random_sample(2);
        assert_bits("interleave_post", c.seed, &post, &c.interleave_post_bits);
    }
}
