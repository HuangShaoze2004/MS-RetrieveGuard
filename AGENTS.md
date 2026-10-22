# MS-RetrieveGuard working rules

- Protect existing work. Do not modify other environments or manage other users' processes.
- Develop directly on `main` and push to `origin/main`. The owner uses one branch; do not create feature branches for routine work unless explicitly requested.
- Calibration and test are sealed for the first fourteen development days. Identity, split, duplicate, candidate eligibility, and exposure audits are allowed; model scores, embeddings, retrieval metrics, selector fitting, and threshold tuning on those parts are forbidden.
- Group queries by standardized two-dimensional molecular identity. All spectra and candidate variants of a query molecule stay in one role. Never count repeated spectra, candidate variants, or model seeds as independent calibration samples.
- Present/absent pairs have equal K. Remove every ground-truth-equivalent identity from absent pools. Filter insufficient pools consistently and report retention. Keep formula and measured-mass information conditions separate.
- Do not inherit upstream automatic test evaluation, fingerprint-equality labels, candidate-zero labels, or random epsilon tie-breaking. Use explicit molecular identities and deterministic ties independent of candidate order.
- Zero acceptance means undefined selective risk, not zero. Unknown FAR and prediction-set coverage have separate denominators. Do not extend calibration claims to arbitrary shifts or unknown prevalence.
- Use project-local environments, caches, temporary files, data, and weights under artifacts/ on the data disk. Invoke their Python by absolute path. Do not commit downloaded data, weights, secrets, or environments.
- Only one GPU job at a time. First implementation package: <=2 GPU wall-hours, <=6 GB downloads, <=20 GiB project disk. First two weeks: <=20 GPU wall-hours, <=15 GB downloads, <=80 GiB. Stop before exceeding these caps; task launch is not completion.
- Save configuration, source/data revisions, checksums, molecule/candidate manifests, raw results, logs, failures, process status, and measured resources. Distinguish paper claims, static implementation, reachable artifacts, successful execution, and numerical reproduction.
- Implement the smallest useful pipeline. Robust selectors require evidence that strong simple baselines leave a substantive gap. Preserve negative results.
- Subagents may work in parallel on code, source audits, and statistics; coordinate file ownership and keep GPU execution serial.
