#!/usr/bin/env Rscript
# Reproducible R ChainLadder parity & timing harness for reserving-kernels.
# Requires: R package `ChainLadder` (>= 0.2.21, the version cited in README).
#
# Every parity value is asserted against the reserving-kernels reference below
# within an explicit relative tolerance; the script exits non-zero on any
# mismatch. Timings are informational only and include R interpreter overhead.
#
# Usage:
#   Rscript scripts/parity_r.R [--skip-timings]

if (!requireNamespace("ChainLadder", quietly = TRUE)) {
  message("ERROR: R package `ChainLadder` is not installed (install.packages('ChainLadder')).")
  quit(status = 2)
}
if (utils::packageVersion("ChainLadder") < "0.2.21") {
  message("ERROR: ChainLadder >= 0.2.21 required, found ", utils::packageVersion("ChainLadder"))
  quit(status = 2)
}
suppressPackageStartupMessages(library(ChainLadder))
cat(sprintf("ChainLadder %s\n", as.character(utils::packageVersion("ChainLadder"))))

# Reference values produced by reserving-kernels (and asserted in
# tests/source_benchmarks.rs). Each entry: c(expected, relative tolerance).
reference <- list(
  mack_raa_reserve        = c(52135.23, 1e-6),
  mack_raa_se             = c(26909.01, 1e-5),
  mw_raa_cdr_se           = c(25181.95, 1e-5),
  mack_genins_reserve     = c(18680855.61, 1e-6),
  mack_genins_se          = c(2447094.86, 1e-5),
  # Clark (2003) loglogistic LDF on GenIns (no truncation); crate-derived.
  clark_ll_genins_reserve = c(35534710.25, 1e-3),
  # ODP GLM (England & Verrall 2002) on GenIns.
  glm_odp_genins_reserve  = c(18680855.61, 1e-6),
  glm_odp_genins_se       = c(2945646.0, 1e-4)
)

failures <- character(0)
check <- function(key, actual) {
  ref <- reference[[key]]
  rel_err <- abs(actual - ref[1]) / abs(ref[1])
  ok <- is.finite(rel_err) && rel_err <= ref[2]
  cat(sprintf(
    "[%s] %-24s actual=%.2f expected=%.2f rel_err=%.2e tol=%.0e\n",
    if (ok) "OK  " else "FAIL", key, actual, ref[1], rel_err, ref[2]
  ))
  if (!ok) failures <<- c(failures, key)
}

# Evaluate the unevaluated expression afresh on every repetition (a plain
# `force()` would only evaluate the promise once and time a cached value).
bench_us <- function(expr, reps = 20L) {
  expr <- substitute(expr)
  env <- parent.frame()
  times <- numeric(reps)
  for (i in seq_len(reps)) {
    t0 <- proc.time()[["elapsed"]]
    eval(expr, env)
    times[i] <- (proc.time()[["elapsed"]] - t0) * 1e6
  }
  median(times)
}

cat("=== Numerical Parity (mages/ChainLadder) ===\n")
mack_raa <- MackChainLadder(RAA, est.sigma = "Mack")
check("mack_raa_reserve", sum(mack_raa$FullTriangle[, ncol(RAA)] - getLatestCumulative(RAA)))
check("mack_raa_se", mack_raa$Total.Mack.S.E)

cdr_raa <- CDR(mack_raa)
check("mw_raa_cdr_se", cdr_raa["Total", "CDR(1)S.E."])

mack_gen <- MackChainLadder(GenIns, est.sigma = "Mack")
check("mack_genins_reserve", sum(mack_gen$FullTriangle[, ncol(GenIns)] - getLatestCumulative(GenIns)))
check("mack_genins_se", mack_gen$Total.Mack.S.E)

clark_gen <- ClarkLDF(GenIns, G = "loglogistic")
check("clark_ll_genins_reserve", clark_gen$Total$FutureValue)

glm_gen <- glmReserve(GenIns, var.power = 1)
check("glm_odp_genins_reserve", glm_gen$summary["Total", "IBNR"])
# Column is labelled "S.E" or "S.E." depending on ChainLadder release.
se_col <- grep("^S\\.E", colnames(glm_gen$summary), value = TRUE)[1]
check("glm_odp_genins_se", glm_gen$summary["Total", se_col])

if (!("--skip-timings" %in% commandArgs(trailingOnly = TRUE))) {
  cat("\n=== Median Single-Threaded Timings (include interpreter overhead) ===\n")
  cat(sprintf("Mack Chain Ladder (RAA 10x10):       %.1f us\n", bench_us(MackChainLadder(RAA, est.sigma = "Mack"), 50L)))
  cat(sprintf("Merz-Wuthrich 1-Yr CDR (RAA):        %.1f us\n", bench_us(CDR(MackChainLadder(RAA, est.sigma = "Mack")), 50L)))
  cat(sprintf("Clark Loglogistic MLE (GenIns):      %.1f us\n", bench_us(ClarkLDF(GenIns, G = "loglogistic"), 30L)))
  cat(sprintf("ODP GLM Reserve (GenIns, IRLS):      %.1f us\n", bench_us(glmReserve(GenIns, var.power = 1), 30L)))
  cat(sprintf("ODP Bootstrap (GenIns, 1k sims):     %.2f ms\n", bench_us(BootChainLadder(GenIns, R = 1000L, process.distr = "od.pois"), 5L) / 1e3))
}

if (length(failures) > 0) {
  message("\nPARITY FAILED: ", paste(failures, collapse = ", "))
  quit(status = 1)
}
cat("\nAll parity checks passed.\n")
