//! Worked example: Mack (1993) Chain-Ladder on the RAA (1991) 10x10 triangle.
//!
//! Run with:
//! ```bash
//! cargo run --example mack_raa
//! ```
//!
//! Companion write-up: `docs/examples/mack-chain-ladder-raa.md`
//! Primary source: T. Mack (1993), "Distribution-Free Calculation of the
//! Standard Error of Chain Ladder Reserve Estimates", ASTIN Bulletin 23(2),
//! pp. 213-225.

use reserving_kernels::datasets;
use reserving_kernels::diagnostics::mack_tests::{
    adjacent_factor_correlation_test, calendar_year_effect_test,
};
use reserving_kernels::triangle::latest_diagonal;
use reserving_kernels::validate::ValidationError;

const ORIGIN_YEARS: [u32; 10] = [1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990];

// Published reserves and standard errors from Mack (1993), Table 2 ($000s).
const MACK_1993_RESERVE: [f64; 10] = [
    0.0, 154.0, 617.4, 1_636.1, 2_746.7, 3_649.1, 5_435.3, 10_907.2, 10_650.0, 16_339.4,
];
const MACK_1993_SE: [f64; 10] = [
    0.0, 206.2, 623.4, 747.2, 1_469.5, 2_001.9, 2_209.2, 5_357.9, 6_333.2, 24_566.3,
];
const MACK_1993_TOTAL_RESERVE: f64 = 52_135.2;
const MACK_1993_TOTAL_SE: f64 = 26_909.0;

fn main() -> Result<(), ValidationError> {
    let raa = datasets::raa();
    let view = raa.view();

    // Mack (1993) distribution-free chain-ladder with unit tail factor (1.0).
    // `mack()` defaults to `AllowNegativeIncrements` because incurred triangles
    // (including RAA origin 1982, dev 6 -> 7: 15,599 -> 15,496) can exhibit
    // minor negative development.
    let mack = raa.mack(1.0)?;
    let latest = latest_diagonal(&view);

    println!("Mack (1993) Chain-Ladder Worked Example — RAA (1991) 10x10 ($000s)");
    println!("===================================================================\n");

    println!("1. Development Factors (f_k) and Variance Parameters (sigma_k^2)");
    println!("-------------------------------------------------------------------");
    println!("{:<10} {:>12} {:>16}  Note", "Link", "f_k", "sigma_k^2");
    for k in 0..mack.ldf.len() {
        let note = if k + 1 == mack.ldf.len() {
            "extrapolated via Mack min-rule"
        } else {
            ""
        };
        println!(
            "{:<10} {:>12.4} {:>16.2}  {note}",
            format!("{} -> {}", k + 1, k + 2),
            mack.ldf[k],
            mack.sigma_sq[k],
        );
    }

    println!("\n2. Origin-Level & Aggregate Reserves vs. Mack (1993) Table 2 ($000s)");
    println!("-------------------------------------------------------------------");
    println!(
        "{:<6} {:>9} {:>10} {:>10} {:>10} {:>7} | {:>10} {:>10}",
        "Origin", "Latest", "Ultimate", "Reserve", "Mack SE", "CV", "Paper R", "Paper SE"
    );

    let mut total_latest = 0.0;
    let mut total_ultimate = 0.0;
    let mut sum_origin_mse = 0.0;
    for i in 0..ORIGIN_YEARS.len() {
        total_latest += latest[i];
        total_ultimate += mack.ultimate[i];
        sum_origin_mse += mack.standard_error[i] * mack.standard_error[i];

        let cv_str = if mack.reserve[i] > 0.0 {
            format!("{:>6.1}%", mack.cv[i] * 100.0)
        } else {
            "      -".to_string()
        };

        println!(
            "{:<6} {:>9.1} {:>10.1} {:>10.1} {:>10.1} {:>7} | {:>10.1} {:>10.1}",
            ORIGIN_YEARS[i],
            latest[i],
            mack.ultimate[i],
            mack.reserve[i],
            mack.standard_error[i],
            cv_str,
            MACK_1993_RESERVE[i],
            MACK_1993_SE[i],
        );
    }

    println!("-------------------------------------------------------------------");
    println!(
        "{:<6} {:>9.1} {:>10.1} {:>10.2} {:>10.2} {:>6.1}% | {:>10.1} {:>10.1}",
        "Total",
        total_latest,
        total_ultimate,
        mack.total_reserve,
        mack.total_standard_error,
        mack.total_cv * 100.0,
        MACK_1993_TOTAL_RESERVE,
        MACK_1993_TOTAL_SE,
    );

    let indep_se = sum_origin_mse.sqrt();
    println!(
        "\nNote: sqrt(sum SE_i^2) = {indep_se:.2} vs. Total Mack SE = {:.2}.",
        mack.total_standard_error,
    );
    println!(
        "The difference reflects positive cross-origin covariance from shared LDF estimation (Mack 1993, Eq. 15)."
    );

    // Going further on the same triangle: Mack (1994) assumption diagnostics
    // and Merz & Wuthrich (2008) 1-year Claims Development Result (Solvency II).
    let corr = adjacent_factor_correlation_test(&view);
    let cal = calendar_year_effect_test(&view);
    let cdr = raa.merz_wuthrich()?;

    println!("\n3. Going Further on RAA: Assumption Diagnostics & 1-Year CDR");
    println!("-------------------------------------------------------------------");
    println!(
        "Mack (1994) adjacent-factor correlation: T = {:+.4}, two-sided p = {:.4}",
        corr.statistic_t, corr.two_sided_p_value,
    );
    println!(
        "Mack (1994) calendar-year effect test:   Z = {:+.4}, two-sided p = {:.4}",
        cal.statistic_z, cal.two_sided_p_value,
    );
    println!(
        "Merz-Wuthrich (2008) 1-year CDR SE:      {:.2} ({:.1}% of ultimate Mack SE)",
        cdr.total_cdr_standard_error,
        (cdr.total_cdr_standard_error / mack.total_standard_error) * 100.0,
    );

    Ok(())
}
