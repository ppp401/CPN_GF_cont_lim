# Legacy pipeline

These scripts are frozen numerical references for the former simulate → flow →
plot workflow. They still read the old `gf_data/` and `gf_results/` schemas but
are not maintained as part of the active pipeline. Run them from the repository
root if an old experiment must be reproduced.

Their NumPy samplers, flow implementation, and statistics helpers live in
`legacy/func/`. Active code must not import those modules except in parity tests.
