#!/usr/bin/env python3
"""Prepare LJSpeech or VCTK audio and write deterministic JSONL manifests."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import soundfile as sf
import soxr
import yaml
from tqdm import tqdm


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Utterance:
    utterance_id: str
    speaker_id: str
    text: str
    audio_path: Path


def repository_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping")

    for section in ("dataset", "audio", "preprocessing"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing configuration section: {section}")

    dataset_name = str(config["dataset"].get("name", "")).lower()
    if dataset_name not in {"ljspeech", "vctk"}:
        raise ValueError("dataset.name must be 'ljspeech' or 'vctk'")

    sample_rate = int(config["audio"].get("sample_rate", 0))
    if sample_rate <= 0:
        raise ValueError("audio.sample_rate must be a positive integer")

    validation_fraction = float(config["preprocessing"].get("validation_fraction", 0.0))
    test_fraction = float(config["preprocessing"].get("test_fraction", 0.0))
    if validation_fraction < 0 or test_fraction < 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("Validation and test fractions must be non-negative and total less than one")

    for key in ("root",):
        if not config["dataset"].get(key):
            raise ValueError(f"Missing dataset.{key}")
    for key in ("output_dir", "manifest_dir"):
        if not config["preprocessing"].get(key):
            raise ValueError(f"Missing preprocessing.{key}")
    return config


def ljspeech_utterances(dataset: dict) -> Iterator[Utterance]:
    root = repository_path(str(dataset["root"]))
    metadata_path = root / str(dataset.get("metadata_file", "metadata.csv"))
    audio_dir = root / str(dataset.get("audio_dir", "wavs"))
    delimiter = str(dataset.get("metadata_delimiter", "|"))
    text_column = int(dataset.get("text_column", 2))
    extension = str(dataset.get("audio_extension", ".wav"))
    speaker_id = str(dataset.get("speaker_id", "ljspeech"))

    with metadata_path.open("r", encoding="utf-8") as stream:
        for line_number, row in enumerate(csv.reader(stream, delimiter=delimiter), start=1):
            if len(row) <= text_column:
                raise ValueError(f"Invalid metadata row {line_number}: expected column {text_column}")
            utterance_id = row[0].strip()
            yield Utterance(
                utterance_id=utterance_id,
                speaker_id=speaker_id,
                text=row[text_column].strip(),
                audio_path=audio_dir / f"{utterance_id}{extension}",
            )


def vctk_utterances(dataset: dict) -> Iterator[Utterance]:
    root = repository_path(str(dataset["root"]))
    audio_glob = str(dataset.get("audio_glob", "wav48_silence_trimmed/*/*_mic1.flac"))
    transcript_dir = root / str(dataset.get("transcript_dir", "txt"))
    transcript_extension = str(dataset.get("transcript_extension", ".txt"))

    for audio_path in sorted(root.glob(audio_glob)):
        speaker_id = audio_path.parent.name
        transcript_id = re.sub(r"_mic\d+$", "", audio_path.stem)
        transcript_path = transcript_dir / speaker_id / f"{transcript_id}{transcript_extension}"
        if not transcript_path.is_file():
            raise FileNotFoundError(f"Missing transcript for {audio_path}: {transcript_path}")
        yield Utterance(
            utterance_id=audio_path.stem,
            speaker_id=speaker_id,
            text=transcript_path.read_text(encoding="utf-8").strip(),
            audio_path=audio_path,
        )


def split_utterances(items: list[Utterance], config: dict) -> dict[str, list[Utterance]]:
    preprocessing = config["preprocessing"]
    shuffled = list(items)
    random.Random(int(preprocessing.get("seed", 1337))).shuffle(shuffled)

    validation_count = round(len(shuffled) * float(preprocessing.get("validation_fraction", 0.01)))
    test_count = round(len(shuffled) * float(preprocessing.get("test_fraction", 0.01)))
    validation_end = validation_count
    test_end = validation_end + test_count
    return {
        "validation": shuffled[:validation_end],
        "test": shuffled[validation_end:test_end],
        "train": shuffled[test_end:],
    }


def convert_audio(source: Path, destination: Path, audio_config: dict) -> tuple[int, int]:
    samples, source_rate = sf.read(source, dtype="float32", always_2d=True)
    if bool(audio_config.get("mono", True)):
        samples = samples.mean(axis=1)
    if source_rate != int(audio_config["sample_rate"]):
        samples = soxr.resample(samples, source_rate, int(audio_config["sample_rate"]))
    samples = np.asarray(samples, dtype=np.float32)

    if bool(audio_config.get("peak_normalize", False)):
        peak = float(np.max(np.abs(samples))) if samples.size else 0.0
        if peak > 0.0:
            samples = 0.99 * samples / peak

    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destination, samples, int(audio_config["sample_rate"]), subtype="PCM_16")
    return int(samples.shape[0]), int(audio_config["sample_rate"])


def output_audio_path(output_dir: Path, item: Utterance, dataset_name: str) -> Path:
    if dataset_name == "vctk":
        return output_dir / item.speaker_id / f"{item.utterance_id}.wav"
    return output_dir / f"{item.utterance_id}.wav"


def portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPOSITORY_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def prepare(config: dict, limit: int | None) -> None:
    dataset_name = str(config["dataset"]["name"]).lower()
    iterator = ljspeech_utterances(config["dataset"]) if dataset_name == "ljspeech" else vctk_utterances(config["dataset"])
    utterances = list(iterator)
    if limit is not None:
        utterances = utterances[:limit]
    if not utterances:
        raise ValueError("No utterances were found")

    missing = [item.audio_path for item in utterances if not item.audio_path.is_file()]
    if missing:
        preview = "\n".join(str(path) for path in missing[:5])
        raise FileNotFoundError(f"Missing {len(missing)} audio files. First paths:\n{preview}")

    output_dir = repository_path(str(config["preprocessing"]["output_dir"]))
    manifest_dir = repository_path(str(config["preprocessing"]["manifest_dir"]))
    manifest_dir.mkdir(parents=True, exist_ok=True)

    for split_name, items in split_utterances(utterances, config).items():
        manifest_path = manifest_dir / f"{split_name}.jsonl"
        with manifest_path.open("w", encoding="utf-8") as manifest:
            for item in tqdm(items, desc=split_name):
                destination = output_audio_path(output_dir, item, dataset_name)
                sample_count, sample_rate = convert_audio(item.audio_path, destination, config["audio"])
                record = {
                    "audio_path": portable_path(destination),
                    "utterance_id": item.utterance_id,
                    "speaker_id": item.speaker_id,
                    "text": item.text,
                    "sample_rate": sample_rate,
                    "num_samples": sample_count,
                    "duration_seconds": round(sample_count / sample_rate, 6),
                }
                manifest.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Prepared {len(utterances)} utterances in {output_dir}")
    print(f"Wrote manifests to {manifest_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Dataset YAML configuration")
    parser.add_argument("--limit", type=int, help="Process only the first N discovered utterances")
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Validate and summarize the configuration without accessing the dataset",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    config = load_config(args.config)
    if args.check_config:
        print(f"Configuration is valid: {args.config}")
        print(f"Dataset: {config['dataset']['name']}")
        print(f"Target sample rate: {config['audio']['sample_rate']} Hz")
        return
    prepare(config, args.limit)


if __name__ == "__main__":
    main()
