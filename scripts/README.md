# `scripts/`

One-off utility, not part of the regular workflow.

- `remove_stale_noise_rows.py` — used once, when the `low`/`medium` noise probabilities were
  corrected to match the assignment's guideline table exactly. It removed the handful of
  already-saved result rows that had been computed under the old (pre-correction) noise values,
  so the experiment runner would redo just those rows instead of skipping them as "already done".
  Kept here for a transparent record of that correction; there's no need to run it again.
