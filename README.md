# MELD Multimodal Emotion Recognition

A work-in-progress prototype that combines utterance text and audio features to classify emotions using MELD. The project focuses on a small, understandable pipeline with simple baseline models and an eventual interaction demo.

## Project Goal

Compare three approaches:

- **Text-only:** TF-IDF features with Logistic Regression.
- **Audio-only:** MFCC summary features with Logistic Regression.
- **Text + audio:** combine both modalities and compare against the individual baselines.

The final prototype is intended to accept utterances over time and return a structured emotion prediction with a short response grounded in the input.

## Current Progress

The audio preprocessing pipeline has been implemented and checked:

- Match media files to CSV records using `Dialogue_ID` and `Utterance_ID`.
- Convert MP4 files to mono, 16 kHz WAV using FFmpeg.
- Skip existing outputs and record conversion failures.
- Extract 13 MFCC coefficients and summarize each with its mean and standard deviation.
- Build an audio feature matrix while keeping features and emotion labels aligned.

The latest local train extraction produced:

| Item | Result |
| --- | --- |
| Audio feature matrix | `(9988, 26)` |
| Emotion labels | `(9988,)` |
| Unavailable sample | `dia125_utt3.wav` |

The corresponding MP4, `dia125_utt3.mp4`, could not be parsed by FFmpeg and was skipped. These numbers describe preprocessing coverage, not classification performance.

Feature caching, dev preprocessing, audio model training, multimodal integration, and the interaction demo are still in progress.

## Audio Features

Each utterance becomes a vector of 26 values:

- The first 13 values are the means of the MFCC coefficients across time.
- The remaining 13 values are their standard deviations across time.

For `N` usable utterances, the resulting matrix has shape `(N, 26)`. Labels come from the matching CSV rows; feature extraction does not use emotion labels.

## Main Audio Scripts

| File | Purpose |
| --- | --- |
| `audio_features.py` | Extract a 26-dimensional feature vector from one WAV file |
| `4_prepare_audio.py` | Convert train MP4 files to WAV and handle missing or failed inputs |
| `5_inspect_audio_features.py` | Extract train features and check array shapes and label alignment |

The numbered filenames reflect the current development workflow.

## Setup

Use a Python environment with the following packages:

```bash
python -m pip install numpy pandas librosa scikit-learn
```

FFmpeg must also be installed and available from the terminal:

```bash
ffmpeg -version
```

Download MELD separately. Place the training videos at:

```text
data/raw/MELD.Raw/train/train_splits/
```

Set the train CSV path in the scripts to the location of your MELD CSV. The audio output directory is:

```text
data/audio/train/
```

Dataset files and generated outputs are excluded from Git through `.gitignore`.

## Run the Audio Pipeline

Run these commands from the project root.

### 1. Convert training videos to WAV

```bash
python 4_prepare_audio.py
```

The script checks for missing inputs, skips existing WAV files, and continues after recorded FFmpeg conversion failures.

### 2. Check one audio feature vector

```bash
python audio_features.py
```

The current example uses `data/audio/train/dia0_utt0.wav`. Expected feature shape:

```text
(26,)
```

### 3. Extract train audio features

```bash
python 5_inspect_audio_features.py
```

Expected output structure:

```text
X_audio shape: (N, 26)
y_audio shape: (N,)
```

The latest local run used 9,988 samples. Your count may differ if your media files are missing or unreadable. The current extraction script builds arrays in memory; saving them is the next step.

## Evaluation Plan

- Train models and fit feature transformations using the train split.
- Use the dev split for model comparison and limited hyperparameter tuning.
- Use the test split after selecting the final configuration.
- Report Macro-F1, accuracy, and per-class results.
- Compare modalities on the same usable samples.
- Retain CSV row indices alongside cached audio features so text, audio, and labels stay aligned after skipped samples.

## Next Steps

- [ ] Save train audio features, labels, and CSV row indices.
- [ ] Prepare and extract dev audio features.
- [ ] Train and evaluate the audio-only baseline.
- [ ] Integrate text and audio features.
- [ ] Complete final test evaluation.
- [ ] Add structured outputs and a small sequential interaction demo.
- [ ] Document results and example predictions.

## Scope and Limitations

This is an educational prototype in active development. Real-time interaction and a complete multimodal inference pipeline have not yet been implemented.

MFCC means and standard deviations summarize each clip and do not preserve its detailed temporal sequence. Emotion predictions are dataset category estimates, rather than direct measurements of a person's internal emotional state.

## Dataset

This project uses the **Multimodal EmotionLines Dataset (MELD)**. Dataset media and annotations are obtained separately and are not included in this repository.

