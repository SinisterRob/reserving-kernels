#!/usr/bin/env Rscript
# Reproducible R ChainLadder parity & timing harness for reserving-kernels.
# Requires: R package `ChainLadder` (>= 0.2.19).
#
# Usage:
#   Rscript scripts/parity_r.R

suppressPackageStartupMessages(library(ChainLadder))

bench_us <- function(expr, reps = 20L) {
  times <- numeric(reps)
  for (i in seq_len(reps)) {
    t0 <- proc.time()[["elapsed"]]
    force(expr)
    times[i] <- (proc.time()[["elapsed"]] - t0) * 1e6
  }
  median(times)
}

cat("=== Numerical Parity (mages/ChainLadder) ===\n")
mack_raa <- MackChainLadder(RAA, est.sigma = "Mack")
cat(sprintf(
  "Mack RAA:    reserve = %.2f, SE = %.2f\n",
  sum(mack_raa$FullTriangle[, ncol(RAA)] - getLatestCumulative(RAA)),
  mack_raa$Total.Mack.S.E
))

cdr_raa <- CDR(mack_raa)
cat(sprintf("MW 1-yr RAA: CDR SE  = %.2f\n", cdr_raa["Total", "CDR(1)S.E."]))

mack_gen <- MackChainLadder(GenIns, est.sigma = "Mack")
cat(sprintf(
  "Mack GenIns: reserve = %.2f, SE = %.2f\n",
  sum(mack_gen$FullTriangle[, ncol(GenIns)] - getLatestCumulative(GenIns)),
  mack_gen$Total.Mack.S.E
))

clark_gen <- ClarkLDF(GenIns, G = "loglogistic")
cat(sprintf(
  "Clark LL GenIns: reserve = %.2f, SE = %.2f\n",
  clark_gen$Total$FutureValue,
  clark_gen$Total$StdError
))

glm_gen <- glmReserve(GenIns, var.power = 1)
cat(sprintf(
  "ODP GLM GenIns:  reserve = %.2f, SE = %.2f\n",
  glm_gen$summary["Total", "IBNR"],
  glm_gen$summary["Total", "S.E."]
))

cat("\n=== Median Single-Threaded Timings ===\n")
cat(sprintf("Mack Chain Ladder (RAA 10x10):       %.1f us\n", bench_us(MackChainLadder(RAA, est.sigma = "Mack"), 50L)))
cat(sprintf("Merz-Wuthrich 1-Yr CDR (RAA):        %.1f us\n", bench_us(CDR(MackChainLadder(RAA, est.sigma = "Mack")), 50L)))
cat(sprintf("Clark Loglogistic MLE (GenIns):      %.1f us\n", bench_us(ClarkLDF(GenIns, G = "loglogistic"), 30L)))
cat(sprintf("ODP GLM Reserve (GenIns):            %.1f us\n", bench_us(glmReserve(GenIns, var.power = 1), 30L)))
cat(sprintf("ODP Bootstrap (GenIns, 1k sims):     %.2f ms\n", bench_us(BootChainLadder(GenIns, R = 1000L, process.distr = "od.pois"), 5L) / 1e3))
