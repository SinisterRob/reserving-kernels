# Mack Chain-Ladder in Rust: A Worked Example

This note reproduces the canonical **Mack (1993)** distribution-free chain-ladder reserve and standard error calculation on the **Reinsurance Association of America (RAA, 1991)** $10 \times 10$ incurred loss triangle using [`reserving-kernels`](https://github.com/SinisterRob/reserving-kernels) (`v0.3.0`).

Every figure below is computed directly by the runnable example [`examples/mack_raa.rs`](https://github.com/SinisterRob/reserving-kernels/blob/main/examples/mack_raa.rs) and verified in CI by [`mack_1993_raa_matches_published_reserves_and_standard_errors`](https://github.com/SinisterRob/reserving-kernels/blob/main/tests/source_benchmarks.rs#L162-L240).

---

## 1. Source Paper & Dataset

- **Method:** T. Mack (1993), *"Distribution-Free Calculation of the Standard Error of Chain Ladder Reserve Estimates"*, *ASTIN Bulletin*, Vol. 23, No. 2, pp. 213–225. DOI: [`10.2143/AST.23.2.2005092`](https://doi.org/10.2143/AST.23.2.2005092).
- **Dataset:** Reinsurance Association of America (1991), Automatic Facultative General Liability (excluding asbestos & environmental), cumulative incurred losses in **\$000s** (Mack 1993, Table 1; compiled into [`src/datasets.rs`](https://github.com/SinisterRob/reserving-kernels/blob/main/src/datasets.rs#L33-L50)).

### Input Triangle (`$000s`)

| Origin | Dev 1 | Dev 2 | Dev 3 | Dev 4 | Dev 5 | Dev 6 | Dev 7 | Dev 8 | Dev 9 | Dev 10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **1981** | 5,012 | 8,269 | 10,907 | 11,805 | 13,539 | 16,181 | 18,009 | 18,608 | 18,662 | 18,834 |
| **1982** | 106 | 4,285 | 5,396 | 10,666 | 13,782 | 15,599 | 15,496 | 16,169 | 16,704 | |
| **1983** | 3,410 | 8,992 | 13,873 | 16,141 | 18,735 | 22,214 | 22,863 | 23,466 | | |
| **1984** | 5,655 | 11,555 | 15,766 | 21,266 | 23,425 | 26,083 | 27,067 | | | |
| **1985** | 1,092 | 9,565 | 15,836 | 22,169 | 25,955 | 26,180 | | | | |
| **1986** | 1,513 | 6,445 | 11,702 | 12,935 | 15,852 | | | | | |
| **1987** | 557 | 4,020 | 10,946 | 12,314 | | | | | | |
| **1988** | 1,351 | 6,947 | 13,112 | | | | | | | |
| **1989** | 3,133 | 5,395 | | | | | | | | |
| **1990** | 2,063 | | | | | | | | | |

> **Note on negative development:** Origin 1982 exhibits a slight negative incurred increment from Dev 6 to Dev 7 (`15,599 → 15,496`). `Triangle::mack` defaults to `MonotonicityPolicy::AllowNegativeIncrements` so incurred and reinsurance triangles run without extra boilerplate, while `Triangle::mack_with_policy(..., MonotonicityPolicy::Strict)` enforces non-decreasing cumulatives when desired.

---

## 2. Reproducible Rust Code

Add `reserving-kernels` to `Cargo.toml`:

```toml
[dependencies]
reserving-kernels = "0.3"
```

Run Mack's chain-ladder directly on the built-in RAA benchmark triangle (with unit tail factor `1.0`):

```rust
use reserving_kernels::datasets;
use reserving_kernels::validate::ValidationError;

fn main() -> Result<(), ValidationError> {
    let raa = datasets::raa();
    let mack = raa.mack(1.0)?;

    println!(
        "Total reserve: {:.2}, Mack SE: {:.2}, CV: {:.1}%",
        mack.total_reserve,
        mack.total_standard_error,
        mack.total_cv * 100.0,
    );
    Ok(())
}
```

For your own data, construct a triangle from upper-row slices without manual `NaN` padding via `Triangle::try_from_upper_rows(&[&[...], ...])?`. To reproduce every table in this document from the repository, run:

```bash
cargo run --example mack_raa
```

---

## 3. Estimated Development Factors & Variance Parameters

Under Mack (1993), volume-weighted age-to-age factors $\hat{f}_k$ and link variance parameters $\hat{\sigma}_k^2$ are:

$$\hat{f}_k = \frac{\sum_{i=0}^{n-k-2} C_{i,k+1}}{\sum_{i=0}^{n-k-2} C_{i,k}}, \qquad \hat{\sigma}_k^2 = \frac{1}{n - k - 2} \sum_{i=0}^{n-k-2} C_{i,k} \left(\frac{C_{i,k+1}}{C_{i,k}} - \hat{f}_k\right)^2$$

For the final development link ($9 \to 10$), only one observed pair exists, so $\hat{\sigma}_8^2$ is extrapolated via Mack's min-rule:

$$\hat{\sigma}_8^2 = \min\!\left(\frac{\hat{\sigma}_7^4}{\hat{\sigma}_6^2},\; \hat{\sigma}_7^2,\; \hat{\sigma}_6^2\right) = \min\!\left(\frac{7.88^2}{1.34},\; 7.88,\; 1.34\right) = 1.34$$

| Development Link | $\hat{f}_k$ | $\hat{\sigma}_k^2$ | Note |
| --- | ---: | ---: | --- |
| 1 → 2 | 2.9994 | 27,883.48 | |
| 2 → 3 | 1.6235 | 1,108.53 | |
| 3 → 4 | 1.2709 | 691.44 | |
| 4 → 5 | 1.1717 | 61.23 | |
| 5 → 6 | 1.1134 | 119.44 | |
| 6 → 7 | 1.0419 | 40.82 | |
| 7 → 8 | 1.0333 | 1.34 | |
| 8 → 9 | 1.0169 | 7.88 | |
| 9 → 10 | 1.0092 | 1.34 | Extrapolated via Mack min-rule |

---

## 4. Reserve & Standard Error vs. Mack (1993) Table 2

Per-origin MSE (Mack 1993, Theorem 3 / Eq. 7) combines process variance and parameter estimation variance; aggregate MSE (Eq. 15) adds positive cross-origin covariance from shared development factors:

| Origin | Latest | Ultimate | Reserve (`reserving-kernels`) | Mack SE (`reserving-kernels`) | CV | Mack (1993) Table 2 Reserve | Mack (1993) Table 2 SE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **1981** | 18,834.0 | 18,834.0 | 0.0 | 0.0 | — | 0.0 | 0.0 |
| **1982** | 16,704.0 | 16,858.0 | 154.0 | 206.2 | 133.9% | 154.0 | 206.2 |
| **1983** | 23,466.0 | 24,083.4 | 617.4 | 623.4 | 101.0% | 617.4 | 623.4 |
| **1984** | 27,067.0 | 28,703.1 | 1,636.1 | 747.2 | 45.7% | 1,636.1 | 747.2 |
| **1985** | 26,180.0 | 28,926.7 | 2,746.7 | 1,469.5 | 53.5% | 2,746.7 | 1,469.5 |
| **1986** | 15,852.0 | 19,501.1 | 3,649.1 | 2,001.9 | 54.9% | 3,649.1 | 2,001.9 |
| **1987** | 12,314.0 | 17,749.3 | 5,435.3 | 2,209.2 | 40.6% | 5,435.3 | 2,209.2 |
| **1988** | 13,112.0 | 24,019.2 | 10,907.2 | 5,357.9 | 49.1% | 10,907.2 | 5,357.9 |
| **1989** | 5,395.0 | 16,045.0 | 10,650.0 | 6,333.2 | 59.5% | 10,650.0 | 6,333.2 |
| **1990** | 2,063.0 | 18,402.4 | 16,339.4 | 24,566.3 | 150.3% | 16,339.4 | 24,566.3 |
| **Total** | **160,987.0** | **213,122.2** | **52,135.23** | **26,909.01** | **51.6%** | **52,135.2** | **26,909.0** |

Notice that $\sqrt{\sum_i \text{SE}_i^2} = 26{,}160.18 < 26{,}909.01$: ignoring cross-origin parameter covariance understates aggregate reserve uncertainty by `$748.8k` on RAA.

---

## 5. Going Further on the Same Triangle

Running Mack (1994) assumption diagnostics and Merz–Wüthrich (2008) 1-year Claims Development Result (Solvency II) on the same RAA triangle (included in [`examples/mack_raa.rs`](https://github.com/SinisterRob/reserving-kernels/blob/main/examples/mack_raa.rs)):

| Diagnostic / Model | Result on RAA | Interpretation |
| --- | --- | --- |
| **Mack (1994) Adjacent-Factor Correlation** | $T = +0.0696$, two-sided $p = 0.7128$ | No statistically significant serial correlation between consecutive link ratios. |
| **Mack (1994) Calendar-Year Effect Test** | $Z = +14.0000$, two-sided $p = 0.5727$ | No statistically significant calendar-diagonal shock pattern. |
| **Merz–Wüthrich (2008) 1-Year CDR SE** | `25,181.95` (`93.6%` of ultimate Mack SE) | Most reserve uncertainty resolves in the next 12 months, dominated by origin 1990 developing from age 1 to 2. |

---

## 6. Traceability & Verification

- **Implementation:** [`src/mack.rs`](https://github.com/SinisterRob/reserving-kernels/blob/main/src/mack.rs) (Mack 1993 Eq. 7 & Eq. 15; Mack 1999 Eq. 4 stochastic tail)
- **Built-in Dataset:** [`src/datasets.rs`](https://github.com/SinisterRob/reserving-kernels/blob/main/src/datasets.rs#L33-L50) (`datasets::raa()`)
- **Runnable Example:** [`examples/mack_raa.rs`](https://github.com/SinisterRob/reserving-kernels/blob/main/examples/mack_raa.rs)
- **Automated Parity Test:** [`tests/source_benchmarks.rs`](https://github.com/SinisterRob/reserving-kernels/blob/main/tests/source_benchmarks.rs#L162-L240) (`mack_1993_raa_matches_published_reserves_and_standard_errors`)
- **Full Literature Map:** [`docs/SOURCES.md`](https://github.com/SinisterRob/reserving-kernels/blob/main/docs/SOURCES.md)
