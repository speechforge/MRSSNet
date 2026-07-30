# MRSSNet

This public repository contains the dataset setup shared by MRSSNet experiments.

The initial release is intentionally limited to:

- dataset directory conventions;
- LJSpeech configurations for 16 kHz and 24 kHz audio;
- a VCTK configuration for 24 kHz audio; and
- a generic manifest and audio preprocessing script.

Model architecture, training and inference implementations, checkpoints, generated
audio, experiment results, and research notes are not included.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Place a locally obtained dataset under `data/` using the layout in
[`docs/DATASETS.md`](docs/DATASETS.md). Dataset contents are ignored by Git.

Check a configuration without reading or writing audio:

```bash
python scripts/prepare_dataset.py \
  --config configs/datasets/ljspeech_24khz.yaml \
  --check-config
```

Prepare a dataset and create deterministic train, validation, and test manifests:

```bash
python scripts/prepare_dataset.py \
  --config configs/datasets/ljspeech_24khz.yaml
```

Use `ljspeech_16khz.yaml` for the 16 kHz LJSpeech variant or `vctk_24khz.yaml`
for VCTK.

## Data policy

No dataset audio, transcripts, manifests generated from local data, or derived
features are committed. Obtain each dataset from its official distributor and
follow its license and usage terms.
