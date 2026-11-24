# Knee Osteoarthritis Severity Classification under Class Imbalance

XBDS3024N Image Processing and Computer Vision – individual assignment
**Author:** Syahirah Binti Shamsudin

This project classifies knee X-rays into five Kellgren–Lawrence (KL) grades (KL0–KL4). It compares four models and four ways of handling class imbalance. The dataset has 13 healthy knees (KL0) for every severe knee (KL4).

**Live dashboard:** _add your Streamlit link here_

## Models and techniques
| | |
|---|---|
| Classical models | Linear SVM and Random Forest on HOG features |
| Neural models | MobileNetV2 and DenseNet121 (frozen ImageNet backbone + trained head) |
| Imbalance handling | None, class weighting, random oversampling, augmentation oversampling |
| Selection | Highest validation macro-F1; test set used only for reporting |

## Key results (test set, 1,656 images)
- Best single models: accuracy 0.47–0.51, macro-F1 0.40–0.46.
- A soft-voting ensemble (MobileNetV2 + DenseNet121 + SVM) reached accuracy 0.540 and macro-F1 0.490.
- A binary control experiment (KL0 vs KL1–4) reached accuracy 0.70–0.71. The low five-class accuracy comes mainly from neighbouring KL grades looking almost identical.

## Repository contents
| File / folder | Contents |
|---|---|
| `IPCV_Knee_OA_4Models_with_S15.ipynb` | Main notebook: data, preprocessing, features, 20 experiments, evaluation, overfitting analysis (Section 15) |
| `IPCV_Knee_OA_Binary_Check.ipynb` | Binary control experiment (KL0 vs KL1–4, KL0–1 vs KL2–4) |
| `knee_oa_dashboard.py` | Streamlit dashboard for radiologists (Home, Analysis, Prediction) |
| `results/` | Saved metrics and trained models used by the dashboard |
| `Knee_Dataset/test/` | Test X-rays for the dashboard's examples and "Challenge the AI" game |

## Run the dashboard locally
```bash
pip install -r requirements.txt
python -m streamlit run knee_oa_dashboard.py
```

## Dataset
[Knee Osteoarthritis Dataset with Severity Grading](https://www.kaggle.com/datasets/shashwatwork/knee-osteoarthritis-dataset-with-severity) (Kaggle). Only the test split is included here. Download the full dataset from Kaggle to re-run the notebooks.

> Coursework prototype for decision support only. It is not a medical device and does not provide a diagnosis.
