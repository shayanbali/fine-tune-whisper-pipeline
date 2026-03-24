# Multi-Model Whisper Fine-Tuning Pipeline

This project provides a clean and scalable pipeline for fine-tuning and inference multiple OpenAI Whisper models on custom speech datasets using Hugging Face Transformers.

The script supports training several models sequentially on the same dataset and saving each fine-tuned model separately.


## Features

- Fine-tune multiple Whisper models in one run
- Shared dataset split for fair comparison
- Automatic saving of processor + model
- Reproducible configuration saved per run
- Mixed precision (fp16) training support
- Gradient accumulation support


## 📊 Dataset Format

Your dataset CSV must contain:

| Column | Description |
|--------|-------------|
| audio  | Path to audio file |
| text   | Transcription |

Example:

```csv
audio,text
audio/0001.wav,Hello world
audio/0002.wav,How are you
```


## ⚙️ Configuration

### Data Configuration

```python
data_config = DataConfig(
    csv_path="/path/to/metadata.csv",
    audio_column="audio",
    text_column="text",
    sampling_rate=16000,
    test_size=0.1,
)
```


### Model List (Multiple Fine-Tuning)

Add as many models as you want:

```python
model_configs = [
    ModelConfig(
        model_name="openai/whisper-small",
        output_dir="outputs/whisper-small",
    ),
    ModelConfig(
        model_name="openai/whisper-base",
        output_dir="outputs/whisper-base",
    ),
]
```

Each model will be trained and saved independently.


Training will run sequentially for each model.


## 💾 Output

Each model directory will contain:

```
outputs/
└── whisper-small/
    ├── Checkpoints
    ├── pytorch_model.bin
    ├── model.safetensors
    ├── config.json
    ├── preprocessor_config.json
    ├── tokenizer.json
    └── run_config.json
```

