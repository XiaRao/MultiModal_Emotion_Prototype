# MELD Multimodal Emotion Recognition

An utterance-level text-and-audio emotion recognition prototype using MELD. It combines TF-IDF text features and MFCC audio summaries with Logistic Regression, returns structured emotion predictions and template-based replies, and demonstrates sequential processing of recorded clips.

## Approach

- **Text:** TF-IDF with up to 5,000 unigram/bigram features.
- **Audio:** 13 MFCC means and 13 standard deviations per utterance.
- **Fusion:** standardize audio features and concatenate them with TF-IDF features. The fitted fusion model uses 5,026 input features.
- **Classifier:** Logistic Regression with `max_iter=1000` and `class_weight="balanced"`.
- **Replies:** hand-written templates selected by the predicted emotion.

Media files are matched to annotations using `Dialogue_ID` and `Utterance_ID`. Cached audio features retain their original CSV row indices so text, audio, and labels remain aligned after skipped samples.

The seven output categories are anger, disgust, fear, joy, neutral, sadness, and surprise.

## Results

### Dev comparison

All three models below use the same 9,988 usable train utterances and 1,108 usable dev utterances. The Text-only reference uses the same fitted TF-IDF features as the fusion model.

| Model | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| Text-only, balanced | 37.82% | 30.52% |
| Audio-only, balanced | 21.21% | 18.35% |
| Text + Audio, balanced | **41.43%** | **34.73%** |

Fusion improved dev Macro-F1 by 4.21 percentage points over the Text-only reference. An initial Audio-only model without class weighting achieved 44.86% accuracy and 15.32% Macro-F1; the balanced version was retained because it achieved higher dev Macro-F1.

### Final test evaluation

The saved fusion model was evaluated with its selected settings and fitted preprocessing components. All 2,610 test utterances were usable.

| Metric | Result |
| --- | ---: |
| Accuracy | 40.46% |
| Macro-F1 | 31.59% |

| Emotion | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| anger | 0.295 | 0.380 | 0.332 | 345 |
| disgust | 0.093 | 0.221 | 0.130 | 68 |
| fear | 0.056 | 0.180 | 0.085 | 50 |
| joy | 0.368 | 0.413 | 0.389 | 402 |
| neutral | 0.691 | 0.410 | 0.515 | 1,256 |
| sadness | 0.259 | 0.375 | 0.306 | 208 |
| surprise | 0.412 | 0.505 | 0.454 | 281 |

For context, always predicting neutral would achieve 48.12% test accuracy and approximately 9.28% Macro-F1. The fusion model has lower accuracy than that strategy but recognizes a broader set of emotions. Fear and disgust remain weak categories.

The reported improvement over Text-only is a **dev finding**; only the fusion model has been evaluated on test in this workflow.

## Data Coverage

| Split | CSV records | Usable audio samples | Skipped sample |
| --- | ---: | ---: | --- |
| Train | 9,989 | 9,988 | `dia125_utt3.wav` |
| Dev | 1,109 | 1,108 | `dia110_utt7.wav` |
| Test | 2,610 | 2,610 | None |

The train MP4 `dia125_utt3.mp4` failed FFmpeg conversion. The dev WAV `dia110_utt7.wav` was unavailable during extraction.

## Setup

The local workflow was run on macOS with Python 3.14. Install the Python packages in your project environment:

```bash
python -m pip install numpy pandas scipy librosa scikit-learn joblib
```

Install FFmpeg separately and verify that it is available:

```bash
ffmpeg -version
```

Run scripts from the project root. Obtain MELD separately and use the following paths, or adjust the scripts to your local extraction layout:

| Input | Path |
| --- | --- |
| Train annotations | `data/annotations/train_sent_emo.csv` |
| Dev annotations | `data/annotations/dev_sent_emo.csv` |
| Test annotations | `data/annotations/test_sent_emo.csv` |
| Train MP4s | `data/raw/MELD.Raw/train/train_splits/` |
| Dev MP4s | `data/raw/MELD.Raw/dev/dev_splits_complete/` |
| Test MP4s | `data/raw/MELD.Raw/test/output_repeated_splits_test/` |

Generated WAV files are stored in `data/audio/{train,dev,test}/`. Features are stored in `data/features/`, and the fitted model bundle is stored in `data/models/`. The `data/` directory is excluded from Git; another user must prepare the data and train the model before running inference.

## Workflow

### Prepare audio and cache features

The conversion scripts produce mono, 16 kHz WAV files, skip existing outputs, and record conversion failures. The extraction scripts use the shared `audio_features.py` function and save `X_audio`, `y_audio`, and `row_indices` in split-specific `.npz` files.

```bash
python 4_prepare_audio.py
python 5_inspect_audio_features.py
python 6_prepare_audio.py
python 7_inspect_audio_features.py
python 12_prepare_test_audio.py
python 13_extract_test_audio_features.py
```

### Train and compare models

```bash
python 2_train_text.py
python 8_train_audio.py
python 9_trian_multimodal.py
```

The legacy `2_train_text.py` uses the full text splits. The matched-sample Text-only comparison reported above is trained inside the fusion script.

The fusion script saves `data/models/multimodal_model.joblib`, containing:

- `vectorizer`: fitted vocabulary and IDF weights.
- `scaler`: fitted audio feature means and standard deviations.
- `model`: fitted fusion classifier.

The spelling of `9_trian_multimodal.py` reflects the current script filename.

### Predict a single utterance

```bash
python inference.py
```

This loads the model bundle and demonstrates prediction on the first dev utterance. The module also exposes:

```python
predict_emotion(text, audio_path)
```

The function returns an emotion string and a template-based response. Ground-truth labels are used only by the demonstration code for comparison.

### Sequential demo

```bash
python 11_stream_demo.py
```

The demo processes the first ten dev utterances in CSV order and prints one JSON event per input, containing:

- Dialogue and utterance IDs, speaker, and input text.
- Predicted emotion and response.
- Time received relative to the demo start.
- Processing time in milliseconds.

A one-second pause between events simulates successive arrivals. It does not reproduce the original conversation timing.

One local ten-clip run measured 394.31 ms for the first prediction and 3.25–8.94 ms for the following nine, with a mean of 5.88 ms for those nine. These illustrative measurements cover WAV loading, feature extraction, and prediction; they exclude initial model loading, MP4 conversion, microphone capture, and the artificial pause.

### Final test evaluation

```bash
python 14_evaluate_test.py
```

This aligns test text with cached audio features, applies the saved preprocessing with `transform()`, and evaluates the saved classifier with `predict()`. It does not refit preprocessing or train the classifier on test.

## Scope and Limitations

- The demo processes completed, prerecorded utterances sequentially; live microphone streaming has not been implemented.
- Each prediction uses the current utterance only, without conversational memory.
- MFCC means and standard deviations discard detailed temporal order. Fusion joins utterance-level features without aligning individual words to audio frames.
- Speaker identity is annotation metadata, and event timing describes the replay. Neither is inferred by the classifier.
- Replies are generic emotion-conditioned templates. They do not perform open-ended dialogue generation or extract topic-specific content.
- A wrong emotion prediction can select an inappropriate reply. The first dev example predicts surprise for an utterance annotated as sadness.
- Emotion categories are dataset predictions, not direct measurements of a person's internal emotional state.

## Dataset

This project uses the **Multimodal EmotionLines Dataset (MELD)**. Media and annotations are obtained separately and are not redistributed in this repository.
