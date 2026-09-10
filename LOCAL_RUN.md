# Local analysis workflow

This repository is now intended to use GitHub primarily for version control. The heavy pilot and work-space GitHub Actions workflows are manual-only.

## 1. Clone and install

```bash
git clone https://github.com/HaSuNo0620/irui-analysis.git
cd irui-analysis
git checkout analysis-pipeline
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For downstream analysis only, PyTorch/model downloads are not needed once cached embeddings are present.

## 2. One-time semantic cache bootstrap

If GitHub CLI (`gh`) is installed and authenticated:

```bash
bash bootstrap_local_cache.sh
```

This downloads the already-produced run-33 artifact. It does not start a GitHub Actions run.

The downstream runner requires:

```text
results/candidates.csv
results/chunk_manifest.csv
results/embeddings_raw.npy
results/embeddings_anonymized.npy
```

These four files can also be copied into `results/` manually.

## 3. Run all lightweight downstream analyses

```bash
python run_analysis.py
```

The default sequence is:

1. work-level semantic representations
2. continuous geometry diagnostics
3. common-component residualization
4. residual-axis extreme-work/chunk-position extraction
5. compact machine-readable summary

The final compact summary is written to:

```text
results/local_summary.json
```

This file is deliberately small so it can be shared with ChatGPT together with selected CSV/PNG outputs for interpretation without rerunning the expensive embedding stage.

## 4. Configuration

Edit `local_analysis.yaml` to control the lightweight local workflow. The current primary residualization is `remove_common_pcs: 3`, with raw and anonymized variants interpreted in parallel.

You can skip a stage temporarily:

```bash
python run_analysis.py --skip work_space --skip geometry
```

## Resource model

Normal iteration should be:

```text
cached embeddings -> local CPU analysis -> results/*.csv + local_summary.json
```

The Syosetu711K corpus and Ruri encoder are not accessed by `run_analysis.py`. GitHub Actions and Codex are therefore unnecessary for ordinary downstream iterations.

## Future control-corpus stage

A random Syosetu711K control corpus should be cached separately rather than mixed into the selected corpus, e.g.

```text
data/selected/
data/random_control/
```

The intended comparison is the geometry of the selected human/nonhuman relationship corpus relative to general web-novel variation, not a supervised classifier using the original selection lexicons.
