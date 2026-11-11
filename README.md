# enterprise-network

Code, logs and firm-level data for a study of **degree bias in link prediction on the ownership network of Korean listed firms**, built from public disclosures retrieved through OpenDART (Financial Supervisory Service, Korea).

The accompanying paper is *When Degree Debiasing Meets Zero-Degree Nodes: As-of Link Prediction and Personnel Interlocks in a Corporate Ownership Network* (under submission). It predicts which listed firm an investor will newly take a stake in next year, under as-of rules (every input filed by the prediction date), with a separate zero-degree group, no-graph controls, 10-seed ensembles, and a preregistered confirmatory test on FY2025→26.

## Where to look

| What | Where |
|---|---|
| Preregistered confirmatory protocol (amendment A8, frozen 2026-11-08) | `linkpred/protocol_confirm.md`, between `A8 BEGIN` and `A8 END`, at tag `protocol-frozen-A8`; report-only amendments at tags `protocol-amend-A8-1` and `protocol-amend-A8-2` |
| Every development analysis, with its design commit (written before running) and results commit | `reports/DEV_RESULTS.md` |
| Data construction and the observation-gap audit | `reports/DATA_V3.md`, `linkpred/src/build_v3.py`, `linkpred/src/check_v3.py` |
| Exploratory test year 2024→25 (earlier preregistration A7) and its corrections | `reports/PAPER_RESULTS.md`, `linkpred/protocol.md` |
| Models and evaluation | `linkpred/src/paper_dev.py`, `paper_models.py`, `paper_data.py`, `baselines.py` |
| Confirmatory run script (frozen) | `linkpred/src/paper_confirm.py` |
| Controlled-isolation experiments on public benchmarks | `linkpred/bench/` |
| Backtest 2016–2018 | `linkpred/src/backcast.py`, `backcast_fix.py`; `reports/DEV_RESULTS.md` §26.5–26.10 |

Earlier stages of the project (descriptive statistics, node-level prediction, the first link-prediction stages) are documented in `reports/` and `docs/`; the original stage-A README is `docs/README_stage_A_descriptive.md`. Most reports are written in Chinese; the paper and its supplementary material are in English.

## Layout

- `src/`: data download and processing for the descriptive and node-level stages (OpenDART client, panels, edges).
- `linkpred/`: the link-prediction study. `src/` code; `data_v3x/` the firm-level data used in the paper (candidate pairs, labels, yearly graphs, eligibility); `data_v3b/` the 2014–2019 backtest data; `out_*/` evaluation outputs (score files are not tracked).
- `reports/`: analysis logs and reports. `reruns/`: robustness reruns of the descriptive stage. `docs/`: API notes and code walkthroughs.

## Data

All data derive from OpenDART. Its terms of use do not grant an explicit redistribution license; matters they do not regulate fall under Korea's Copyright Act and Public Data Act. The tracked files contain code and firm-level derived data (DART corporate codes, candidate pairs, labels, the evidence used to check that each tie is new, and pair-level counts such as the number of shared directors). They contain no person-level data such as names or birth months of directors and officers. Readers can rebuild everything with their own API key, read from the environment variable `OPENDART_API_KEY` (never written to code, logs or caches).

## Reproducing

Python 3.12. The frozen environment for the confirmatory run uses numpy 2.5.4, pandas 3.1.0, scikit-learn 1.9.1, scipy 1.18.1, torch 2.15.0 and PyTorch Geometric 2.8.1 (exact versions in `linkpred/out_confirm_dryrun/manifest.json`).

```bash
cd linkpred
python src/build_v3.py --out data_v3x --first-year 2019   # needs data/interim from the raw-data release
python src/check_v3.py                                    # ten unit checks
python src/paper_dev.py --help                            # development runs and evaluation
```

The confirmatory test on FY2025→26 will be run once, after April 30, 2027, with the frozen code at commit `3e7c1cb`.

## Contact

Mingsen Zhang (open an issue in this repository).
