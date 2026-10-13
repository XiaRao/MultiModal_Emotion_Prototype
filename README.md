## MELD Multimodal Emotion Recognition Prototype Summary

- The chosen track is **Text and Audio**.
- **Logistic Regression** is used for learning and makign predictions.
- **real time** is defined as the model returning the prediction of the emotion and the intended time for short response in 500ms after both inputs are available.
  - Its text and corresponding mono, 16kHz WAV is already available and the model bundle is loaded before interaction begins.
- A reasonable minimum workflow for this prototype is: Receive the first text + audio message (wav) → Use TF-IDF and MFCC to process the text and audio accordingly → standardize → feature fusing → use the logistic regression model to predict → Return emotion + short reply → Receive the next input → Return the result again.
- 
- **limitations:**
  - The short response is pre-structured depending on the emotion, so the response might not be grounded enough that aligns with the actual context provided by the user.
  -  different input under the same emotion category always receive the same response. The model does not adjust its response with respect to the context.
  - The demo processes completed, prerecorded utterances sequentially; live microphone streaming has not been implemented.
  - MFCC means and standard deviations discard detailed temporal order. This means the temporal order of words don't contribute to the prediction process of the model, which might be a significant factor that helps making decision about the emotion. Considering other approaches to deal with MFCC might increase the model prediction accuracy.
  - A wrong emotion prediction can select an inappropriate reply. The first dev example predicts surprise for an utterance annotated as sadness. The accuracy is low.
  - Predicting the speaker's emotion from solely the text and audio of speech might lack sufficient evidence. Since spoken words cannot comprehensively represent the speaker's inner state. To make better predictions, building dialogue history and user profile might help.

## Approach

- **Text:** TF-IDF with up to 5,000 unigram/bigram features.
- **Audio:** 13 MFCC means and 13 standard deviations per utterance.
- **Fusion:** standardize audio features and concatenate them with TF-IDF features. The fitted fusion model uses 5,026 input features.
- **Classifier:** Logistic Regression with `max_iter=1000` and `class_weight="balanced"`.
- **Replies:** handwritten templates selected by the predicted emotion.

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
python -m pip install -r requirements.txt
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

## Follow-up Experiments and Reflection
Two follow-up experiments investigated whether coarse audio timing or a different linear classifier would improve the model performance. Both used the same utterances and the same TFIDF settings.

| Experiment | Audio features | Classifier | Dev accuracy | Dev Macro-F1 |
| --- | ---: | --- | ---: | ---: |
| Original baseline | 26 | Logistic Regression, balanced | 41.43% | 34.73% |
| Second experiment: three-segment MFCC | 78 | Logistic Regression, balanced | 40.70% | 33.94% |
| Third experiment: LinearSVC | 26 | Balanced, C=1.0 | 43.32% | 32.51% |
| Third experiment: LinearSVC | 26 | Balanced, C=0.1 | **45.49%** | **35.35%** |
| Third experiment: LinearSVC | 26 | Balanced, C=10.0 | 40.61% | 29.78% |

### Second Experiment
This experiment divided the MFCC frame sequence into 3 segments, each part contributed the means and standard deviations of 13 coefficients. so it's 78 features for audio part. 
- Dev accuracy decreased by 0.72 percentage and macro-f1 by 0.79 percentage relative to the original model performance.
- The implementation does not establish that temporal information is helpful.
- Possible Explanations:
  - **Less stable summaries:** each segment contains fewer frames than the full utterance. Short clips may therefore produce noisier means and standard deviations.
  - **Limited new information:** segments still compress the MFCC heavily and do not explicitly measure pitch, energy, rhythm, etc. that might contribute to emotion predictions.
### Third Experiment
This experiment returned to the original 26-dimensional audio features, but changed the classification model from LinearRegression model to LinearSVC model. all 3 models used the class_weight = "balanced". However, the model had shown overfitting characteristics when checking the summarized data as follows. At C = 0.1, it improved dev accuracy by 4.06 percentage and macro-f1 by 0.62 percentage. Therefore, this experiment produced a modest improvement than the original one.

| C | Train accuracy | Train Macro-F1 | Dev accuracy | Dev Macro-F1 |
| ---: | ---: | ---: | ---: | ---: |
| 0.1 | 68.96% | 64.63% | **45.49%** | **35.35%** |
| 1.0 | 83.33% | 82.18% | 43.32% | 32.51% |
| 10.0 | 90.10% | 89.51% | 40.61% | 29.78% |

- Larger C -> Weaker regularization. The training performance rose as C increased, while the dev performance fell. This pattern is consistent with overfitting, particularly at C = 1.0 and c = 10.0.
- Possible reasons for the improvement to be limited:
  - Both the original classifier and the third classifier used the same TF-IDF and MFCC summaries. Therefore this does not restore the lost information during the data processing procedure. 
  - Rare categories have limited training evidence and small dev supports: disgust has 22 dev samples and fear has 40. Class weighting changes the training objective but does not create new examples.


## Hardware, Resources, and Parameter Accounting.

| Item | Report|
| Computer Model and CPU| Macbook Air, Apple M4|
| RAM | 16GB|
| Operating System | MacOS|
| Python | 3.14.2|
| Inference Device| CPU|
| Model bundle size | about 459KiB|
| First Prediction in the reported ten-clip run | 414.36ms|
| Subsequent nine predictions | mean 6.76ms|

### External Resources
- FFmpeg: converts MP4 audio to mono, 16kHz WAV.
- librosa==1.0.0: load audio.
- NumPy==2.5.3: compute MFCC summary features
- pandas==3.0.6/SciPy==1.18.1: align annotations and assemble sparse feature matrices
- scikit-learn==1.9.1: implements TF-IDF, standardization, LogisticRegression, and evaluation metrics.
- joblib==1.6.0: persists and reloads the model.
- ChatGPT: assisted with loading data, debugging, and organizing README.md

## Dataset

This project uses the **Multimodal EmotionLines Dataset (MELD)**. Media and annotations are obtained separately and are not redistributed in this repository.

## More Discussions - Current Stopping point
The original Logistic Regression system remains the documented end-to-end baseline. Its test results, latency measurements, memory measurements, and parameter counts apply to that model, not to the experimental SVC model.
The selected SVC candidate was saved separately as data/models/multimodal_model_svc.joblib, with results in data/results/linear_svc_comparison.csv. Deployment of this candidate and measurement of its test performance and runtime are left for a future iteration.
These experiments suggest that additional feature dimensions and a classifier change do not automatically produce large improvements. Stronger regularization helped the tested SVC models, while three-segment pooling did not improve the tested Logistic Regression model. Further work could examine richer speech features, contextual text representations, or dialogue history, but these were left outside the current prototype.
