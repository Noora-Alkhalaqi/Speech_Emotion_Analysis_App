# VoiceSense: Speech Emotion Recognition (Proof of Concept)

A proof of concept that recognizes emotions in short speech clips using classic machine learning. The model predicts one of six emotions from a WAV file, and a small Streamlit web app lets you try it by uploading audio.

> **Status:** early prototype. The results are preliminary and were measured on acted, studio-recorded speech. Performance on real-world audio has not been verified.

## Problem and goal

Understanding emotions from speech is hard, especially across large numbers of recordings, and traditional audio analysis does not identify emotional patterns automatically. The goal is a platform that recognizes emotions from speech and supports emotion-based audio analytics. This proof of concept covers the first step: classifying short clips into six emotions.

## Approach

1. **Data:** CREMA-D audio clips (see below). Labels come from the file names.
2. **Features:** 40 MFCCs per clip (extracted with `librosa`), averaged over time, so each clip becomes a vector of 40 numbers.
3. **Split:** train, validation, and test sets split **by actor**, so the same speaker never appears in both training and evaluation.
4. **Model:** Random Forest (`scikit-learn`).
5. **Demo:** Streamlit app that extracts the same features from an uploaded WAV file and shows the predicted emotion.

## Dataset

This project uses **CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset)**.

- 7,442 clips from 91 actors, 12 sentences, six emotions: Anger, Disgust, Fear, Happy, Neutral, Sad.
- Clips are about 2 seconds long. Audio is processed as mono at 16 kHz.
- Classes are nearly balanced: 1,271 clips per emotion, except Neutral (1,087).
- The emotions are acted, not naturally occurring.
- Kaggle copy used here: [`ejlok1/cremad`](https://www.kaggle.com/datasets/ejlok1/cremad)
- Original repository: [CheyneyComputerScience/CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D)
- License: the dataset is released under the Open Database License (ODbL 1.0), with contents under the Database Contents License (DbCL 1.0). Check the original repository for the exact terms before reuse.

The dataset is **not** included in this repository.

**Citation:**

> Cao H, Cooper DG, Keutmann MK, Gur RC, Nenkova A, Verma R. *CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset.* IEEE Transactions on Affective Computing. 2014;5(4):377-390. doi:10.1109/TAFFC.2014.2336244

```bibtex
@article{cao2014crema,
  title   = {{CREMA-D}: Crowd-sourced emotional multimodal actors dataset},
  author  = {Cao, Houwei and Cooper, David G and Keutmann, Michael K and Gur, Ruben C and Nenkova, Ani and Verma, Ragini},
  journal = {IEEE Transactions on Affective Computing},
  volume  = {5},
  number  = {4},
  pages   = {377--390},
  year    = {2014},
  publisher = {IEEE}
}
```

## Preliminary results

| Metric | Result |
|---|---|
| Majority-class baseline | 18% |
| Validation accuracy | 42% |
| Validation macro F1-score | 0.40 |
| Test accuracy | 43.7% |

- Anger has the highest F1-score (0.63). Sadness has the highest recall (0.74).
- Fear is the most challenging class (F1-score: 0.16). Fear and Disgust are often confused with other emotions.
- For context, the CREMA-D paper reports that human raters recognized the intended emotion from audio alone about 41% of the time.

## Repository structure

```
.
├── app.py                 # Streamlit demo
├── emotion_model.pkl      # trained Random Forest
├── requirements.txt       # Python dependencies
├── [your-notebook].ipynb  # data exploration, features, training, evaluation
└── README.md
```

## How to run the demo

Requires Python 3.12 (the version used during development).

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local address shown in the terminal (usually `http://localhost:8501`), upload a short `.wav` file, and the app shows the predicted emotion.

The model was trained with a specific `scikit-learn` version, so keep the version pinned in `requirements.txt`.

## How to reproduce the training

1. Download the dataset with `kagglehub`:

   ```python
   import kagglehub

   path = kagglehub.dataset_download("ejlok1/cremad")
   ```

2. Run the notebook from top to bottom. It builds the feature table (40 MFCC means per clip), splits the data by actor, trains the Random Forest, evaluates it, and saves `emotion_model.pkl`.

## Limitations

- **Short, acted clips:** CREMA-D contains short clips of actors, while real recordings may be much longer and less clear.
- **Averaging over time:** averaging MFCCs over a whole recording can hide emotional changes.
- **Domain shift:** background noise, accents, microphones, and speaking styles may affect performance. Accuracy on CREMA-D does not guarantee similar performance on real-world audio.
- **One prediction per clip:** it cannot describe how emotion changes during a conversation.

## Next steps

- Tune the Random Forest hyperparameters and compare with neural networks, CNN-based models, and pretrained audio models.
- Analyze long recordings with a sliding window and show an emotion timeline.
- Test on more natural speech and consider additional datasets such as MSP-Podcast.
- Add batch audio analysis, emotion distribution charts, and exportable reports.

## Author

[Your Name] · [Course or program] · [Date]