# A12 paired statistics correction

The original A12 CSVs have 256 measurements per regime but only 128 distinct synthetic spectra: each spectrum is observed under two frozen response matrices. The new `statistical_audit` script reports spectrum-clustered paired Wilcoxon and 20,000-draw bootstrap CIs, Holm-adjusted over 15 tests. The original A12 paired statistics are preserved as historical records; do not treat original sample-level p-values as 256 independent spectra. No new independent physical instrument population is inferred.
