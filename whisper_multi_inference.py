
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any

import librosa
import pandas as pd
import torch
from jiwer import wer, cer
from transformers import WhisperForConditionalGeneration, WhisperProcessor


@dataclass
class DatasetConfig:
    name: str
    json_path: str
    audio_base_dir: Optional[str]
    output_dir: str
    sampling_rate: int = 16000
    language: Optional[str] = None
    task: str = "transcribe"


@dataclass
class ModelConfig:
    name: str
    model_dir: str
    batch_size: int = 1
    max_new_tokens: int = 225


def ensure_dir(path: str | Path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_model_name(name: str):
    return name.replace("/", "__").replace("\\", "__").replace(" ", "_")

# Load dataset
# Customize this section based on your own dataset
def build_audio_path(audio_base_dir: Optional[str], audio_filepath: str) -> str:
    audio_path = Path(audio_filepath)
    if audio_path.is_absolute() or audio_base_dir is None:
        return str(audio_path)
    return str(Path(audio_base_dir) / audio_filepath)


def load_dataset_json(json_path: str) -> List[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if "data" not in payload or not isinstance(payload["data"], list):
        raise ValueError(f"Dataset JSON at {json_path} must contain a top-level 'data' list.")

    return payload["data"]


def load_audio(audio_path: str, sampling_rate: int):
    audio, _ = librosa.load(audio_path, sr=sampling_rate)
    return audio

# Load WER and CER as metrics
def compute_dataset_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    valid_df = df.dropna(subset=["prediction"]).copy()

    if len(valid_df) == 0:
        return {
            "num_samples": int(len(df)),
            "num_valid_predictions": 0,
            "num_failed_samples": int(len(df)),
            "wer": None,
            "cer": None,
        }

    references = valid_df["ground_truth"].astype(str).tolist()
    predictions = valid_df["prediction"].astype(str).tolist()

    return {
        "num_samples": int(len(df)),
        "num_valid_predictions": int(len(valid_df)),
        "num_failed_samples": int(len(df) - len(valid_df)),
        "wer": float(wer(references, predictions)),
        "cer": float(cer(references, predictions)),
    }

# Class for Whisper inference
class WhisperInferenceRunner:
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.processor = WhisperProcessor.from_pretrained(model_config.model_dir)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_config.model_dir)
        self.model.eval()
        self.model.to(self.device)

    def transcribe(
        self,
        audio,
        sampling_rate: int,
        language: Optional[str] = None,
        task: str = "transcribe",
    ):
        inputs = self.processor(
            audio,
            sampling_rate=sampling_rate,
            return_tensors="pt",
        )

        input_features = inputs.input_features.to(self.device)

        generate_kwargs = {
            "max_new_tokens": self.model_config.max_new_tokens,
        }

        if language is not None:
            generate_kwargs["language"] = language
        if task is not None:
            generate_kwargs["task"] = task

        with torch.no_grad():
            predicted_ids = self.model.generate(input_features, **generate_kwargs)

        transcription = self.processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True,
        )[0]

        return transcription.strip()

# Evaluate on single model
def evaluate_single_model_on_dataset(runner: WhisperInferenceRunner, dataset_config: DatasetConfig):
    records = load_dataset_json(dataset_config.json_path)
    results = []

    for i, item in enumerate(records):
        print(f"[{runner.model_config.name}] [{dataset_config.name}] sample {i + 1}/{len(records)}")

        audio_rel_path = item.get("audio_filepath")
        duration = item.get("duration")
        ground_truth = item.get("text")

        result = {
            "voice_id": audio_rel_path,
            "duration": duration,
            "ground_truth": ground_truth,
            "prediction": None,
            "sample_wer": None,
            "sample_cer": None,
            "error": None,
        }

        try:
            audio_path = build_audio_path(dataset_config.audio_base_dir, audio_rel_path)
            audio = load_audio(audio_path, dataset_config.sampling_rate)

            transcription = runner.transcribe(
                audio=audio,
                sampling_rate=dataset_config.sampling_rate,
                language=dataset_config.language,
                task=dataset_config.task,
            )

            result["prediction"] = transcription
            result["sample_wer"] = float(wer(str(ground_truth), transcription))
            result["sample_cer"] = float(cer(str(ground_truth), transcription))

        except Exception as e:
            result["error"] = str(e)

        results.append(result)

    df = pd.DataFrame(results)
    metrics = compute_dataset_metrics(df)

    summary = {
        "model_name": runner.model_config.name,
        "model_dir": runner.model_config.model_dir,
        "dataset_name": dataset_config.name,
        "dataset_json_path": dataset_config.json_path,
        "audio_base_dir": dataset_config.audio_base_dir,
        **metrics,
    }

    return df, summary

# Save outputs and predictions
def save_outputs(
    df: pd.DataFrame,
    summary: Dict[str, Any],
    dataset_config: DatasetConfig,
    model_config: ModelConfig,
):
    dataset_output_dir = ensure_dir(Path(dataset_config.output_dir) / safe_model_name(model_config.name))

    predictions_csv = dataset_output_dir / "predictions.csv"
    summary_json = dataset_output_dir / "summary.json"

    df.to_csv(predictions_csv, index=False)

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved predictions to: {predictions_csv}")
    print(f"Saved summary to: {summary_json}")


def run_inference(
    model_configs: List[ModelConfig],
    dataset_configs: List[DatasetConfig],
    combined_summary_csv: Optional[str] = None,
):
    all_summaries = []

    for model_config in model_configs:
        print("=" * 80)
        print(f"Loading model: {model_config.name}")
        print("=" * 80)

        runner = WhisperInferenceRunner(model_config)

        for dataset_config in dataset_configs:
            print("-" * 80)
            print(f"Running dataset: {dataset_config.name}")
            print("-" * 80)

            df, summary = evaluate_single_model_on_dataset(runner, dataset_config)
            save_outputs(df, summary, dataset_config, model_config)
            all_summaries.append(summary)

    summary_df = pd.DataFrame(all_summaries)

    if combined_summary_csv is not None:
        ensure_dir(Path(combined_summary_csv).parent)
        summary_df.to_csv(combined_summary_csv, index=False)
        print(f"Saved combined summary to: {combined_summary_csv}")

    return summary_df

# Enter model dir based on your local path
if __name__ == "__main__":
    model_configs = [
        ModelConfig(
            name="whisper_model_1",
            model_dir="path1",
            max_new_tokens=225,
        ),
        ModelConfig(
            name="whisper_model_2",
            model_dir="path2",
            max_new_tokens=225,
        ),
    ]

    dataset_configs = [
        DatasetConfig(
            name="testset1",
            json_path="data1.json",
            audio_base_dir="audio1",
            output_dir="output_testset1",
            sampling_rate=16000,
            language=None,
            task="transcribe",
        ),
        DatasetConfig(
            name="testset2",
            json_path="data2.json",
            audio_base_dir="audio2",
            output_dir="output_testset2",
            sampling_rate=16000,
            language=None,
            task="transcribe",
        ),
    ]

    summary_df = run_inference(
        model_configs=model_configs,
        dataset_configs=dataset_configs,
        combined_summary_csv="/mnt/data/all_results_summary.csv",
    )

    print("\nFinal summary:")
    print(summary_df)
