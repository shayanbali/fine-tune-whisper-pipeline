import os
import json
from dataclasses import dataclass, asdict
from typing import List, Optional

import numpy as np
import pandas as pd
from datasets import Audio, Dataset
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


# ============================================================
# Configuration
# ============================================================

@dataclass
class ModelConfig:
    model_name: str
    output_dir: str
    num_train_epochs: int = 10
    learning_rate: float = 1e-4
    per_device_train_batch_size: int = 8
    generation_max_length: int = 225


@dataclass
class DataConfig:
    audio_dir: str
    json_path: str
    sampling_rate: int = 16000
    test_size: float = 0.1
    seed: int = 42
    max_duration: float = 30.0


# ============================================================
# Data utilities
# ============================================================

# Update this section based on your own dataset
def build_dir_name(base_dir: str, filename: str) -> str:
    """
    Build the full audio path based on your existing naming rules.
    """
    filename_splits = filename.split("_")

    if (
        filename_splits[0].startswith("f")
        and (
            filename_splits[1].startswith("I")
            or filename_splits[1].startswith("f")
        )
    ) or filename_splits[0].startswith("o1"):
        gender_dir = "female-voices"
    else:
        gender_dir = "male-voices"

    if filename_splits[1].startswith("I") and not filename_splits[0].startswith("o"):
        speaker_dir = filename_splits[0] + "-bot"
    elif filename_splits[0].startswith("o1"):
        speaker_dir = "o1-f-bot"
    elif filename_splits[0].startswith("o2"):
        speaker_dir = "o2-m-bot"
    elif gender_dir == "female-voices":
        speaker_dir = "f4-bot"
    else:
        speaker_dir = "f4-m-bot"

    return os.path.join(base_dir, gender_dir, speaker_dir, filename)



def load_book_dataframe(data_config: DataConfig) -> pd.DataFrame:
    with open(data_config.json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)["data"]

    df = pd.DataFrame(raw_data)
    df_book = df[df["audio_filepath"].str.contains("book")].copy()

    df_book["audio"] = df_book["audio_filepath"].apply(
        lambda x: build_dir_name(data_config.audio_dir, x)
    )
    df_book = df_book.rename(columns={"text": "sentence"})

    # Keep only rows where audio exists
    df_book = df_book[df_book["audio"].apply(os.path.exists)].copy()
    df_book = df_book[df_book["sentence"].notna()].copy()
    df_book["sentence"] = df_book["sentence"].astype(str)
    df_book = df_book[df_book["sentence"].str.strip() != ""].copy()

    df_book = df_book.reset_index(drop=True)
    return df_book



def prepare_datasets(df: pd.DataFrame, sampling_rate: int, test_size: float, seed: int):
    dataset = Dataset.from_pandas(df)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))
    dataset = dataset.train_test_split(test_size=test_size, seed=seed)
    return dataset["train"], dataset["test"]


# ============================================================
# Data collator
# ============================================================


class WhisperDataCollator:
    def __init__(self, processor, sampling_rate: int = 16000, max_duration: float = 30.0):
        self.processor = processor
        self.sampling_rate = sampling_rate
        self.max_samples = int(sampling_rate * max_duration)

    def __call__(self, features):
        padded_audios = []
        texts = []

        for feature in features:
            array = feature["audio"]["array"]
            text = feature["sentence"]

            if not isinstance(text, str) or not text.strip():
                text = "<unk>"

            if len(array) < self.max_samples:
                pad_width = self.max_samples - len(array)
                array = np.pad(array, (0, pad_width), mode="constant")
            else:
                array = array[: self.max_samples]

            padded_audios.append(array)
            texts.append(text)

        inputs = self.processor(
            padded_audios,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
        )

        labels = self.processor.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=448,
        ).input_ids

        inputs["labels"] = labels
        return inputs


# ============================================================
# Training utilities
# ============================================================


def build_training_args(model_config: ModelConfig) -> Seq2SeqTrainingArguments:
    return Seq2SeqTrainingArguments(
        output_dir=model_config.output_dir,
        per_device_train_batch_size=model_config.per_device_train_batch_size,
        num_train_epochs=model_config.num_train_epochs,
        learning_rate=model_config.learning_rate,
        fp16=True,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        logging_steps=10,
        predict_with_generate=True,
        generation_max_length=model_config.generation_max_length,
        remove_unused_columns=False,
        report_to="none",
    )



def finetune_one_model(
    model_config: ModelConfig,
    train_dataset,
    eval_dataset,
    data_config: DataConfig,
):
    print("\n" + "=" * 80)
    print(f"Finetuning: {model_config.model_name}")
    print(f"Saving to : {model_config.output_dir}")
    print("=" * 80)

    os.makedirs(model_config.output_dir, exist_ok=True)

    processor = AutoProcessor.from_pretrained(model_config.model_name)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_config.model_name)

    data_collator = WhisperDataCollator(
        processor=processor,
        sampling_rate=data_config.sampling_rate,
        max_duration=data_config.max_duration,
    )

    training_args = build_training_args(model_config)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(model_config.output_dir)
    processor.save_pretrained(model_config.output_dir)

    # Save config used for reproducibility
    with open(os.path.join(model_config.output_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(model_config), f, indent=2, ensure_ascii=False)

    print(f"Finished: {model_config.model_name}")



def finetune_multiple_models(
    model_configs: List[ModelConfig],
    data_config: DataConfig,
):
    df_book = load_book_dataframe(data_config)
    print(f"Number of valid book samples: {len(df_book)}")

    train_dataset, eval_dataset = prepare_datasets(
        df=df_book,
        sampling_rate=data_config.sampling_rate,
        test_size=data_config.test_size,
        seed=data_config.seed,
    )

    print(f"Train size: {len(train_dataset)}")
    print(f"Eval size : {len(eval_dataset)}")

    for model_config in model_configs:
        finetune_one_model(
            model_config=model_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_config=data_config,
        )


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":
    # If you are in Colab and need Drive, uncomment these two lines:
    # from google.colab import drive
    # drive.mount('/content/drive')

    data_config = DataConfig(
#     	Set these paths based on your own dataset paths
        audio_dir="/content/drive/MyDrive/shayan/turkey/final_book_dataset/audio",
        json_path="/content/drive/MyDrive/shayan/turkey/final_book_dataset/text/KartalOl_general_voices_0_0_1_training_normalized.json",
        sampling_rate=16000,
        test_size=0.1,
        seed=42,
        max_duration=30.0,
    )

    # Add as many models as you want here.
    # The script will finetune them one by one and save each one separately.
    model_configs = [
        ModelConfig(
            model_name="openai/whisper-small",
            output_dir="/content/drive/MyDrive/shayan/turkey/finetuned_models/whisper_small",
            num_train_epochs=10,
            learning_rate=1e-4,
            per_device_train_batch_size=8,
        ),
        ModelConfig(
            model_name="opeinai/whisper-tiny",
            output_dir="/content/drive/MyDrive/shayan/turkey/finetuned_models/whisper_tiny",
            num_train_epochs=10,
            learning_rate=1e-4,
            per_device_train_batch_size=8,
        ),
    ]

    finetune_multiple_models(
        model_configs=model_configs,
        data_config=data_config,
    )
