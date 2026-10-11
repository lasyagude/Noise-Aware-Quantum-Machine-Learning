# `tests/`

Automated tests for everything in `src/qml_noise/`. 244 tests, all currently passing.

One test file per module, same names:

- `test_config.py`, `test_datasets.py`, `test_classical.py`, `test_circuits.py`, `test_noise.py`,
  `test_vqc.py`, `test_metrics.py`, `test_evaluation.py`, `test_experiments.py`,
  `test_extensions.py`, `test_an_vqc.py`, `test_plotting.py`

## Running them

```powershell
$env:PYTHONPATH="src"
pytest
```

Most tests are fast; a handful actually train a small quantum circuit to check training works
end-to-end, which is why a full run takes a few minutes rather than a few seconds.
