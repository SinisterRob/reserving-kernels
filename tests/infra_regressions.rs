//! Regression tests for crate infrastructure: `Triangle` construction
//! invariants, a deterministic seeded mutation fuzz over `Triangle` JSON, and
//! WASM C-ABI status codes (under `--features wasm`).

use std::panic::{AssertUnwindSafe, catch_unwind};

use reserving_kernels::{ReservingError, Triangle, datasets};

/// Fixed fuzz seed (house default) so every failure is reproducible.
const FUZZ_SEED: u64 = 42069;
/// Number of mutated payloads per seed corpus entry.
const FUZZ_CASES_PER_SEED: usize = 600;

/// SplitMix64: tiny deterministic PRNG, adequate for mutation selection.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, bound: usize) -> usize {
        if bound == 0 {
            return 0;
        }
        usize::try_from(self.next_u64() % (bound as u64)).unwrap_or(0)
    }

    fn pick<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        &items[self.below(items.len())]
    }
}

/// Invariants every successfully constructed `Triangle` must satisfy.
fn assert_triangle_invariants(tri: &Triangle, context: &str) {
    let (no, nd) = (tri.n_origins(), tri.n_devs());
    assert_eq!(
        Some(tri.data().len()),
        no.checked_mul(nd),
        "{context}: shape"
    );
    assert_eq!(no, nd, "{context}: square");
    let n = tri.n();
    for i in 0..n {
        for &v in tri.origin_row(i) {
            assert!(v.is_finite() && v >= 0.0, "{context}: observed cell {v}");
        }
    }
    // Re-validation through the public constructor must agree.
    assert_eq!(
        Triangle::try_new(no, nd, tri.data().to_vec()).as_ref(),
        Ok(tri),
        "{context}: try_new round-trip"
    );
    // Accessors that index the buffer directly must not panic.
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        assert_eq!(tri.latest_diagonal().len(), n);
        assert_eq!(tri.to_incremental().data().len(), n * n);
        let _ = tri.diagnostics();
        let _ = tri.chain_ladder(1.0);
        let _ = tri.mack(1.0);
    }));
    assert!(outcome.is_ok(), "{context}: accessor panicked");
}

fn seed_corpus() -> Vec<String> {
    let small = Triangle::try_from_upper_rows(&[&[100.0, 160.0, 195.0], &[120.0, 190.0], &[135.0]])
        .expect("small triangle");
    [small, datasets::raa(), datasets::genins()]
        .iter()
        .map(|t| serde_json::to_string(t).expect("serialize seed"))
        .collect()
}

/// Byte offsets of every JSON number/null token inside the `data` array.
fn value_spans(json: &str) -> Vec<(usize, usize)> {
    let Some(start) = json.find("\"data\":[") else {
        return Vec::new();
    };
    let body_start = start + "\"data\":[".len();
    let bytes = json.as_bytes();
    let mut spans = Vec::new();
    let mut i = body_start;
    while i < bytes.len() && bytes[i] != b']' {
        let tok_start = i;
        while i < bytes.len() && bytes[i] != b',' && bytes[i] != b']' {
            i += 1;
        }
        spans.push((tok_start, i));
        if i < bytes.len() && bytes[i] == b',' {
            i += 1;
        }
    }
    spans
}

const POISON_VALUES: &[&str] = &[
    "NaN",
    "null",
    "Infinity",
    "-Infinity",
    "1e308",
    "1e400",
    "-1e400",
    "-1.0",
    "-0.0",
    "0",
    "\"7\"",
    "[]",
    "{}",
    "true",
    "18446744073709551616",
];

const POISON_DIMS: &[&str] = &[
    "0",
    "1",
    "2",
    "4",
    "11",
    "-1",
    "1.5",
    "4294967296",
    "18446744073709551615",
    "18446744073709551616",
    "null",
    "\"10\"",
];

fn mutate(rng: &mut SplitMix64, seed: &str) -> Vec<u8> {
    let mut bytes = seed.as_bytes().to_vec();
    match rng.below(7) {
        // Truncation at an arbitrary byte.
        0 => {
            let cut = rng.below(bytes.len() + 1);
            bytes.truncate(cut);
        }
        // Random byte flips (may produce invalid UTF-8).
        1 => {
            for _ in 0..=rng.below(4) {
                let at = rng.below(bytes.len());
                let flip = u8::try_from(rng.next_u64() & 0xFF).unwrap_or(0);
                bytes[at] ^= flip.max(1);
            }
        }
        // Replace a data value with a poison token.
        2 => {
            let spans = value_spans(seed);
            if !spans.is_empty() {
                let (a, b) = *rng.pick(&spans);
                let mut s = String::with_capacity(seed.len() + 8);
                s.push_str(&seed[..a]);
                s.push_str(rng.pick(POISON_VALUES));
                s.push_str(&seed[b..]);
                bytes = s.into_bytes();
            }
        }
        // Shape mismatch: rewrite one or both dimensions.
        3 => {
            let field = *rng.pick(&["n_origins", "n_devs"]);
            let dim = rng.pick(POISON_DIMS);
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(seed) {
                let mut text = v.to_string();
                let key = format!("\"{field}\":");
                if let Some(pos) = text.find(&key) {
                    let vstart = pos + key.len();
                    let vend = text[vstart..]
                        .find([',', '}'])
                        .map_or(text.len(), |o| vstart + o);
                    text.replace_range(vstart..vend, dim);
                }
                bytes = text.into_bytes();
            }
        }
        // Drop or duplicate a data element.
        4 => {
            let spans = value_spans(seed);
            if spans.len() > 1 {
                let (a, b) = *rng.pick(&spans);
                let mut s = seed.to_owned();
                if rng.below(2) == 0 {
                    // Remove the token and its trailing comma when present.
                    let end = if s.as_bytes().get(b) == Some(&b',') {
                        b + 1
                    } else {
                        b
                    };
                    s.replace_range(a..end, "");
                    if s.contains(",]") {
                        s = s.replace(",]", "]");
                    }
                } else {
                    let tok = seed[a..b].to_owned();
                    s.insert_str(b, &format!(",{tok}"));
                }
                bytes = s.into_bytes();
            }
        }
        // Structural damage: remove a field or swap types.
        5 => {
            let variants = [
                seed.replacen("\"data\":", "\"dat\":", 1),
                seed.replacen("\"n_devs\":", "\"n_devz\":", 1),
                seed.replacen('[', "{", 1),
                format!("[{seed}]"),
                format!("{seed}{seed}"),
                seed.replacen('{', "{\"extra\":1,", 1),
            ];
            bytes = rng.pick(&variants).clone().into_bytes();
        }
        // Zero out the whole observed region with a single poison value.
        _ => {
            let poison = rng.pick(POISON_VALUES);
            let mut s = seed.to_owned();
            for (a, b) in value_spans(seed).into_iter().rev() {
                if rng.below(3) == 0 {
                    s.replace_range(a..b, poison);
                }
            }
            bytes = s.into_bytes();
        }
    }
    bytes
}

#[test]
fn seeded_triangle_json_mutation_fuzz_never_panics_and_upholds_invariants() {
    let mut rng = SplitMix64(FUZZ_SEED);
    let corpus = seed_corpus();
    let mut accepted = 0_usize;
    let mut rejected = 0_usize;
    for (s_idx, seed) in corpus.iter().enumerate() {
        // The unmutated seed itself must round-trip.
        let decoded: Triangle = serde_json::from_str(seed).expect("seed decodes");
        assert_triangle_invariants(&decoded, "seed");
        for case in 0..FUZZ_CASES_PER_SEED {
            let payload = mutate(&mut rng, seed);
            let context = format!(
                "seed={FUZZ_SEED} corpus={s_idx} case={case} payload={}",
                String::from_utf8_lossy(&payload)
            );
            let parsed = catch_unwind(|| serde_json::from_slice::<Triangle>(&payload));
            let Ok(result) = parsed else {
                panic!("deserialization panicked: {context}");
            };
            match result {
                Ok(tri) => {
                    accepted += 1;
                    assert_triangle_invariants(&tri, &context);
                }
                Err(_) => rejected += 1,
            }
        }
    }
    // Sanity: the mutator must exercise both outcomes.
    assert!(accepted > 0, "no mutated payload was accepted");
    assert!(rejected > 0, "no mutated payload was rejected");
}

#[test]
fn try_new_rejects_overflowing_dimensions_without_panicking() {
    // 2^32 * 2^32 wraps to 0 on 64-bit targets; must be a ShapeMismatch, not
    // a pass-through that later panics on indexing.
    let huge = 1_usize << (usize::BITS / 2);
    let result = catch_unwind(|| Triangle::try_new(huge, huge, Vec::new()));
    assert_eq!(result.ok(), Some(Err(ReservingError::ShapeMismatch)));
    let json = format!(r#"{{"n_origins":{huge},"n_devs":{huge},"data":[]}}"#);
    let parsed = catch_unwind(|| serde_json::from_str::<Triangle>(&json));
    assert!(matches!(parsed, Ok(Err(_))));
}

#[test]
fn triangle_construction_invariants() {
    // Shape and squareness.
    assert_eq!(
        Triangle::try_new(2, 2, vec![1.0; 3]),
        Err(ReservingError::ShapeMismatch)
    );
    assert_eq!(
        Triangle::try_new(2, 3, vec![1.0; 6]),
        Err(ReservingError::NotSquare)
    );
    // Observed cells must be finite and non-negative; unobserved may be NaN.
    assert_eq!(
        Triangle::try_new(2, 2, vec![1.0, f64::NAN, 3.0, f64::NAN]),
        Err(ReservingError::NotFinite)
    );
    assert_eq!(
        Triangle::try_new(2, 2, vec![1.0, f64::INFINITY, 3.0, f64::NAN]),
        Err(ReservingError::NotFinite)
    );
    assert_eq!(
        Triangle::try_new(2, 2, vec![1.0, -2.0, 3.0, f64::NAN]),
        Err(ReservingError::Negative)
    );
    let ok = Triangle::try_new(2, 2, vec![1.0, 2.0, 3.0, f64::NAN]).expect("valid");
    assert_triangle_invariants(&ok, "try_new");

    // Ragged constructors.
    assert_eq!(
        Triangle::try_from_rows::<Vec<f64>>(&[]),
        Err(ReservingError::ShapeMismatch)
    );
    assert_eq!(
        Triangle::try_from_upper_rows(&[&[1.0, 2.0], &[3.0, 4.0]]),
        Err(ReservingError::ShapeMismatch)
    );
    assert_eq!(
        Triangle::try_from_incremental_rows(&[&[1.0, f64::NAN], &[2.0]]),
        Err(ReservingError::NotFinite)
    );
    let inc = Triangle::try_from_incremental_rows(&[&[1.0, 2.0], &[3.0]]).expect("incremental");
    assert_eq!(inc.get(0, 1), 3.0);
    assert_triangle_invariants(&inc, "incremental");

    // Macro goes through the validated path.
    let tri = reserving_kernels::triangle![[100, 160, 195], [120, 190], [135]].expect("macro");
    assert_triangle_invariants(&tri, "triangle!");

    // Bundled datasets satisfy the public contract.
    for (name, t) in [
        ("raa", datasets::raa()),
        ("genins", datasets::genins()),
        ("quarg_mack.paid", datasets::quarg_mack_2004().paid),
        ("quarg_mack.incurred", datasets::quarg_mack_2004().incurred),
        (
            "friedland.paid",
            datasets::friedland_us_industry_auto().paid,
        ),
        (
            "friedland.reported",
            datasets::friedland_us_industry_auto().reported,
        ),
    ] {
        assert_triangle_invariants(&t, name);
    }
}

/// Name + function pointer for each `rk_ldf_*` export under test.
#[cfg(feature = "wasm")]
type LdfExport = (&'static str, fn(u32) -> i32);

#[cfg(feature = "wasm")]
#[test]
fn wasm_status_codes_distinguish_errors_and_reject_oversized_n() {
    // All WASM assertions live in one test: the exports share process-global
    // buffers, so splitting them would race across libtest threads.
    use reserving_kernels::wasm_exports::{
        RK_STATUS_ERR, rk_bf, rk_bf_expected_from_premium, rk_brosius, rk_cape_cod,
        rk_cape_cod_elr, rk_cdf_compute, rk_chain_ladder, rk_ldf_ex_hi_lo, rk_ldf_ptr,
        rk_ldf_simple_average, rk_ldf_simple_average_last_n, rk_ldf_volume_weighted,
        rk_ldf_volume_weighted_last_n, rk_mack, rk_n_max, rk_prem_ptr, rk_tri_ptr, rk_tri_stride,
        rk_ult_ptr,
    };

    let n_max = rk_n_max();
    let stride = rk_tri_stride() as usize;
    let tri_ptr = rk_tri_ptr();
    let rows: [&[f64]; 4] = [
        &[100.0, 160.0, 195.0, 205.0],
        &[120.0, 190.0, 232.0],
        &[135.0, 218.0],
        &[150.0],
    ];
    // SAFETY: Single test writing within the N_MAX x N_MAX / N_MAX static buffers.
    unsafe {
        for (i, row) in rows.iter().enumerate() {
            for j in 0..4 {
                *tri_ptr.add(i * stride + j) = row.get(j).copied().unwrap_or(f64::NAN);
            }
        }
        for (i, p) in [300.0, 340.0, 380.0, 420.0].into_iter().enumerate() {
            *rk_prem_ptr().add(i) = p;
        }
    }

    let ldf_exports: [LdfExport; 5] = [
        ("volume_weighted", |n| rk_ldf_volume_weighted(n)),
        ("simple_average", |n| rk_ldf_simple_average(n)),
        ("ex_hi_lo", |n| rk_ldf_ex_hi_lo(n)),
        ("vw_last_n", |n| rk_ldf_volume_weighted_last_n(n, 3)),
        ("sa_last_n", |n| rk_ldf_simple_average_last_n(n, 3)),
    ];
    for (name, f) in ldf_exports {
        assert_eq!(f(4), 3, "{name}: n=4");
        assert_eq!(f(1), 0, "{name}: n=1 is success with zero LDFs");
        assert_eq!(f(0), RK_STATUS_ERR, "{name}: n=0");
        assert_eq!(f(n_max + 1), RK_STATUS_ERR, "{name}: n > n_max");
        assert_eq!(f(u32::MAX), RK_STATUS_ERR, "{name}: n = u32::MAX");
        // SAFETY: Reading the cleared N_MAX-element static LDF buffer.
        assert!(unsafe { *rk_ldf_ptr() }.is_nan(), "{name}: LDF cleared");
    }

    // Count exports: >= 1 on success, 0 on invalid n.
    assert_eq!(rk_ldf_volume_weighted(4), 3);
    assert_eq!(rk_cdf_compute(4, 1.0), 4);
    assert_eq!(rk_chain_ladder(4, 1.0), 4);
    assert_eq!(rk_chain_ladder(n_max + 1, 1.0), 0);
    // SAFETY: Reading the cleared static ULT buffer.
    assert!(unsafe { *rk_ult_ptr() }.is_nan());
    assert_eq!(rk_cdf_compute(n_max + 1, 1.0), 0);
    assert_eq!(rk_bf_expected_from_premium(n_max + 1, 0.7), 0);
    assert_eq!(rk_bf_expected_from_premium(4, 0.7), 4);
    assert_eq!(rk_bf(n_max + 1, 1.0), 0);
    assert_eq!(rk_brosius(n_max + 1, 1.0), 0);
    assert_eq!(rk_brosius(0, 1.0), 0);

    // Scalar exports: NaN on invalid n.
    assert!(rk_cape_cod_elr(n_max + 1, 1.0).is_nan());
    assert!(rk_cape_cod(n_max + 1, 1.0).is_nan());
    assert!(rk_mack(n_max + 1, 1.0).is_nan());
    assert!(rk_mack(4, 1.0).is_finite());
}
