# Dataset layouts

Dataset files are local inputs and are not part of this repository. Paths in the
checked-in configurations are relative to the repository root.

## LJSpeech

Both LJSpeech configurations use the standard extracted directory layout:

```text
data/
└── LJSpeech-1.1/
    ├── metadata.csv
    └── wavs/
        ├── LJ001-0001.wav
        ├── LJ001-0002.wav
        └── ...
```

Each `metadata.csv` row is expected to contain the utterance identifier, original
text, and normalized text separated by `|`. The checked-in configuration uses the
normalized text column.

- `ljspeech_16khz.yaml` resamples output audio to 16,000 Hz.
- `ljspeech_24khz.yaml` resamples output audio to 24,000 Hz.

The source dataset is never modified.

## VCTK

The VCTK configuration expects the VCTK 0.92 layout with silence-trimmed audio:

```text
data/
└── VCTK-Corpus-0.92/
    ├── txt/
    │   ├── p225/
    │   │   ├── p225_001.txt
    │   │   └── ...
    │   └── ...
    └── wav48_silence_trimmed/
        ├── p225/
        │   ├── p225_001_mic1.flac
        │   └── ...
        └── ...
```

The default configuration selects microphone 1 recordings, converts them to mono,
and resamples output audio to 24,000 Hz.

## Generated output

The preprocessing script writes converted PCM WAV files below the configured
`output_dir` and JSON Lines manifests below `manifest_dir`:

```text
manifests/<dataset_variant>/
├── train.jsonl
├── validation.jsonl
└── test.jsonl
```

Every manifest row contains a portable relative audio path, utterance identifier,
speaker identifier, transcript, sample rate, sample count, and duration. Generated
audio and manifests are ignored by Git.
