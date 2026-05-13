## Summary
<!-- 1-3 bullets on what changes and why. -->

## Type of change
- [ ] Bug fix
- [ ] New feature
- [ ] Refactor / cleanup
- [ ] Docs
- [ ] CI / tooling

## Test plan
- [ ] `pytest tests/ -v` passes locally
- [ ] `ruff check .` clean
- [ ] (if frontend touched) `npm run lint && npm run build` clean
- [ ] (if pipeline touched) `python3 src/main.py` runs end-to-end on processed data
- [ ] (if API touched) hit `/health` and one inference endpoint locally

## ML-specific checklist (delete if not applicable)
- [ ] No new label-derived feature columns added to training data
- [ ] Walk-forward split still in place; no random shuffle on temporal data
- [ ] SMOTE (if used) is inside the CV `Pipeline`, not applied before split
- [ ] Metrics in `outputs/` look reasonable — no F1 jump from ~0.95 to ~0.99 (smell of leakage)

## Screenshots / output
<!-- Charts, terminal output, or dashboard screenshots if relevant. -->
