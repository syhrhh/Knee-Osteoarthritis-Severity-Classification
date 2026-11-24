"""
🦴 Knee Osteoarthritis Severity Dashboard (Streamlit)
=====================================================
XBDS3024N Image Processing and Computer Vision – application for the assignment

Target user  : Radiologists (the clinicians who grade knee X-rays on the Kellgren–Lawrence scale)
Pages        : Home (introduction) · Analysis (interactive visualisation) · Prediction (upload an X-ray)

The dashboard re-uses what the notebook `IPCV_Knee_OA_4Models.ipynb` saved:
    results/<ID>/metrics.json        metrics of every experiment (E1–E21, F1–F3)
    results/<ID>/model.joblib        trained SVM / Random Forest (+ scaler)
    results/<ID>/model.weights.h5    trained head of MobileNetV2 / DenseNet121
    results/class_counts.csv         images per grade and split
    Knee_Dataset/                    (optional) example X-rays, test images, animated GIF and background

How to run (in the Assignment_IPCV_Knee folder, with the SAME Python that ran the notebook):
    python -m pip install streamlit plotly
    python -m streamlit run knee_oa_dashboard.py

If TensorFlow is not installed in that Python, the dashboard still works with the SVM and
Random Forest, and explains how to switch the neural models on.

DISCLAIMER: experimental coursework model – NOT a medical device and NOT a diagnosis.
"""

from __future__ import annotations

import base64
import importlib.util
import io
import json
import os
import random
import re
import sys
from abc import ABC, abstractmethod
from pathlib import Path

import altair as alt
import cv2
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from skimage.feature import hog

# ==============================================================================================
# Configuration
# ==============================================================================================
APP_DIR = Path(__file__).resolve().parent
RESULTS_DIR = Path(os.environ.get("KNEE_RESULTS_DIR", APP_DIR / "results"))
DATA_DIR = Path(os.environ.get("KNEE_DATA_DIR", APP_DIR / "Knee_Dataset"))
WEIGHTS_DIR = Path(os.environ.get("KERAS_WEIGHTS_DIR", APP_DIR / "weights"))   # only if ImageNet download fails

AUTHOR = "Syahirah Binti Shamsudin"
APP_TITLE = "🦴 Knee OA Severity Dashboard"
TF_AVAILABLE = importlib.util.find_spec("tensorflow") is not None and os.environ.get("KNEE_DISABLE_TF") != "1"

# Background photos (Unsplash, free licence). A blurred mosaic of the dataset's own X-rays is used underneath
# as a fallback, so the page still has a background when there is no internet connection.
BACKGROUNDS = {
    "home": "https://images.unsplash.com/photo-1631563018856-81be9c118283?fm=jpg&q=60&w=1920&auto=format&fit=crop",       # radiologist with patient
    "analysis": "https://images.unsplash.com/photo-1666214280391-8ff5bd3c0bf0?fm=jpg&q=60&w=1920&auto=format&fit=crop",   # clinicians reviewing imaging
    "prediction": "https://images.unsplash.com/photo-1768644675720-2f274f84c87a?fm=jpg&q=60&w=1920&auto=format&fit=crop", # leg anatomy
}

IMAGE_SIZE = 224
CLASS_NAMES = ["KL0", "KL1", "KL2", "KL3", "KL4"]
NUM_CLASSES = len(CLASS_NAMES)
MAIN_EXPERIMENTS = [f"E{i}" for i in range(1, 21)]
MODEL_ORDER = ["SVM", "Random Forest", "MobileNetV2", "DenseNet121"]
NEURAL = {"MobileNetV2", "DenseNet121"}
MODEL_TYPE = {"SVM": "Classical ML", "Random Forest": "Classical ML",
              "MobileNetV2": "Neural network", "DenseNet121": "Neural network"}
MODEL_EMOJI = {"SVM": "📐", "Random Forest": "🌳", "MobileNetV2": "📱", "DenseNet121": "🧠"}
TECHNIQUES = ["none", "class weighting", "random oversampling", "augmentation oversampling"]
GRADE_COLOURS = ["#4C9F70", "#A3C45A", "#E3B23C", "#E07B39", "#C0392B"]   # green → red with severity
DISCLAIMER = ("Decision-support prototype built for coursework. Not a medical device and not a diagnosis – "
              "the radiograph must always be interpreted together with the history and examination.")

KL_INFO = {   # Kellgren & Lawrence (1957)
    "KL0": ("None", "No radiographic features of osteoarthritis."),
    "KL1": ("Doubtful", "Doubtful joint-space narrowing and possible osteophytic lipping."),
    "KL2": ("Minimal", "Definite osteophytes and possible joint-space narrowing."),
    "KL3": ("Moderate", "Multiple osteophytes, definite joint-space narrowing, some sclerosis, possible bone-end deformity."),
    "KL4": ("Severe", "Large osteophytes, marked joint-space narrowing, severe sclerosis and definite bone-end deformity."),
}
NEXT_STEPS = {   # reporting notes for the radiologist, shown after a prediction
    "KL0": ("info", ["No radiographic features of osteoarthritis.",
                     "If symptoms persist, soft-tissue or meniscal pathology may explain them – MRI can be more informative.",
                     "Compare with prior films if available."]),
    "KL1": ("info", ["Doubtful change – the grade with the most inter-reader disagreement.",
                     "Look closely for subtle osteophytic lipping at the tibial spines and joint margins.",
                     "Consider wording such as 'possible early degenerative change' rather than a firm grade."]),
    "KL2": ("info", ["Definite osteophytes; joint space possibly narrowed.",
                     "Note which compartment (medial / lateral) is narrower.",
                     "Weight-bearing views give the most reliable assessment of joint-space width."]),
    "KL3": ("warning", ["Definite narrowing with multiple osteophytes and some sclerosis.",
                        "Describe compartment involvement and alignment (varus / valgus) if visible.",
                        "Compare with prior imaging to document progression."]),
    "KL4": ("warning", ["Marked narrowing, large osteophytes, severe sclerosis and bone-end deformity.",
                        "Bone-on-bone contact and deformity are key information for the referring surgeon.",
                        "Most AI misses at this grade are called KL3 – double-check the joint-space width."]),
}

BACKBONES = {   # name: (keras constructor, preprocess module, offline weights file)
    "MobileNetV2": ("MobileNetV2", "mobilenet_v2", "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5"),
    "DenseNet121": ("DenseNet121", "densenet", "densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5"),
}

USER_ROLE = "Radiologist"          # the dashboard is designed for radiologists who grade knee X-rays
ROLE_EMOJI = "🩻"

st.set_page_config(page_title="Knee OA Severity Dashboard", page_icon="🦴", layout="wide")


# ==============================================================================================
# Layout helpers (same design as the Traffic Accident Risk Dashboard)
# ==============================================================================================
def center_text(text: str, size: int = 30, weight: str = "bold") -> None:
    st.markdown(f"<h1 style='text-align:center; font-size:{size}px; font-weight:{weight}; margin-bottom:0.2rem;'>{text}</h1>",
                unsafe_allow_html=True)


def page_header(page_name: str) -> None:
    center_text("", size=20)
    center_text(APP_TITLE, size=50)
    center_text(page_name, size=30)
    center_text(f"By : {AUTHOR}", size=20)


def set_bg(url: str) -> None:
    """Full-screen background photo + semi-transparent white content card (dataset mosaic as fallback layer)."""
    fallback = background_mosaic()
    layers = f'url("{url}")' + (f', url("data:image/jpeg;base64,{fallback}")' if fallback else "")
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: {layers};
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-position: center;
        }}
        /* Make text readable */
        .block-container, [data-testid="stMainBlockContainer"] {{
            background: rgba(255, 255, 255, 0.86);
            padding: 2rem !important;
            border-radius: 12px;
            margin-top: 2.5rem;
            margin-bottom: 2rem;
        }}
        /* White expanders */
        [data-testid="stExpander"] details {{
            background-color: white !important;
            border-radius: 10px;
        }}
        [data-testid="stExpander"] summary {{ font-weight: 700; }}
        /* Metric cards */
        [data-testid="stMetric"] {{
            background: #ffffff; border: 1px solid #eee; border-radius: 12px; padding: 10px 14px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        }}
        .kl-card {{ padding: 14px 18px; border-radius: 12px; margin-bottom: 0.6rem; }}
        .big-number {{ font-size: 2.4rem; font-weight: 800; line-height: 1.1; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def section(title: str) -> None:
    st.markdown("---")
    center_text(title, size=30)


def tf_help() -> None:
    st.warning(
        "**TensorFlow is not installed in the Python that is running this dashboard**, so the two neural models "
        "(MobileNetV2 and DenseNet121) are switched off and the SVM / Random Forest are used instead.\n\n"
        "**Fix:** start the dashboard with the same Python that ran your notebook (it already has TensorFlow):\n"
        "```\npython -m pip install streamlit plotly\npython -m streamlit run knee_oa_dashboard.py\n```\n"
        f"or install TensorFlow into the Python used now: `\"{sys.executable}\" -m pip install tensorflow`",
        icon="🧩")


# ==============================================================================================
# Data access
# ==============================================================================================
class ResultsRepository:
    """Loads results/<ID>/metrics.json for every experiment saved by the notebook."""

    def __init__(self, results_dir: Path) -> None:
        self.results_dir = results_dir
        self.runs: dict[str, dict] = {}
        self.model_dirs: dict[str, Path] = {}
        self._load()

    def _load(self) -> None:
        if not self.results_dir.is_dir():
            return
        for folder in sorted(self.results_dir.iterdir()):
            file = folder / "metrics.json"
            if folder.is_dir() and file.exists():
                run = json.loads(file.read_text())
                self.runs[run.get("id", folder.name)] = run
                self.model_dirs[run.get("id", folder.name)] = folder
        if "E1" not in self.runs:          # in the notebook E1 re-uses the winning feature-set run (F2)
            for f_id in ("F2", "F1", "F3"):
                run = self.runs.get(f_id)
                if run and run.get("preprocessing") == "baseline" and run["signature"]["representation"].startswith("hog"):
                    self.runs["E1"] = {**run, "id": "E1"}
                    self.model_dirs["E1"] = self.model_dirs[f_id]
                    break

    @property
    def available(self) -> bool:
        return bool(self.runs)

    def family(self, exp_id: str) -> str:
        return self.runs[exp_id].get("family", "")

    def summary(self, ids: list[str] | None = None) -> pd.DataFrame:
        ids = ids or [e for e in MAIN_EXPERIMENTS + ["E21"] if e in self.runs]
        rows = []
        for e in ids:
            r, t = self.runs[e], self.runs[e]["test"]
            rows.append({"Experiment": e, "Model": r.get("family", ""), "Preprocessing": r["preprocessing"],
                         "Imbalance technique": r["imbalance"], "Val macro-F1": r["val"]["macro_f1"],
                         "Accuracy": t["accuracy"], "Macro-F1": t["macro_f1"], "Balanced accuracy": t["balanced_accuracy"],
                         "KL4 sensitivity": t["kl4_recall"], "QWK": t["qwk"], "MAE": t["mae"],
                         "Train–test gap": r["train"]["accuracy"] - t["accuracy"]})
        return pd.DataFrame(rows)

    def best_per_model(self) -> dict[str, str]:
        """Best configuration of each model = highest VALIDATION macro-F1 (same rule as the notebook)."""
        best = {}
        for model in MODEL_ORDER:
            ids = [e for e in MAIN_EXPERIMENTS if e in self.runs and self.family(e) == model]
            if ids:
                best[model] = max(ids, key=lambda e: self.runs[e]["val"]["macro_f1"])
        return best

    def final_model(self) -> str | None:
        ids = [e for e in MAIN_EXPERIMENTS if e in self.runs]
        return max(ids, key=lambda e: self.runs[e]["val"]["macro_f1"]) if ids else None

    def model_file(self, exp_id: str) -> Path | None:
        folder = self.model_dirs.get(exp_id)
        if folder is None:
            return None
        for name in ("model.joblib", "model.weights.h5"):
            if (folder / name).exists():
                return folder / name
        return None

    def usable(self, exp_id: str) -> bool:
        """A saved model exists and its libraries are installed."""
        file = self.model_file(exp_id)
        return file is not None and (file.suffix == ".joblib" or TF_AVAILABLE)

    def prediction_models(self) -> list[str]:
        best = self.best_per_model()
        return [best[m] for m in MODEL_ORDER if m in best and self.usable(best[m])]

    def default_model(self) -> str | None:
        """Final model if it can run here, otherwise the best usable model on validation."""
        usable = self.prediction_models()
        final = self.final_model()
        if final in usable:
            return final
        return max(usable, key=lambda e: self.runs[e]["val"]["macro_f1"]) if usable else None


@st.cache_resource(show_spinner=False)
def get_repository() -> ResultsRepository:
    return ResultsRepository(RESULTS_DIR)


@st.cache_data(show_spinner=False)
def class_counts() -> pd.DataFrame | None:
    csv = RESULTS_DIR / "class_counts.csv"
    if csv.exists():
        df = pd.read_csv(csv, index_col=0)
        return df.loc[[s for s in ("train", "val", "test") if s in df.index], CLASS_NAMES]
    if DATA_DIR.is_dir():
        data = {s: [len(list((DATA_DIR / s / str(g)).glob("*.png"))) for g in range(NUM_CLASSES)]
                for s in ("train", "val", "test") if (DATA_DIR / s).is_dir()}
        if data:
            return pd.DataFrame(data, index=CLASS_NAMES).T
    return None


def split_images(split: str, grade: int) -> list[Path]:
    folder = DATA_DIR / split / str(grade)
    return sorted(folder.glob("*.png")) if folder.is_dir() else []


@st.cache_data(show_spinner=False)
def background_mosaic() -> str | None:
    """Dark, blurred mosaic of training X-rays (base64 JPEG) – offline fallback background."""
    tiles = []
    for g in range(NUM_CLASSES):
        tiles += split_images("train", g)[:8]
    if len(tiles) < 8:
        return None
    random.Random(1).shuffle(tiles)
    imgs = [cv2.resize(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE), (160, 160)) for p in tiles[:40]]
    while len(imgs) < 40:
        imgs += imgs[: 40 - len(imgs)]
    grid = np.vstack([np.hstack(imgs[r * 8:(r + 1) * 8]) for r in range(5)])
    grid = cv2.GaussianBlur(grid, (0, 0), 3)
    colour = cv2.applyColorMap(grid, cv2.COLORMAP_BONE)
    colour = cv2.convertScaleAbs(colour, alpha=0.55, beta=10)
    ok, buf = cv2.imencode(".jpg", colour, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return base64.b64encode(buf).decode() if ok else None


@st.cache_data(show_spinner=False)
def grade_journey_gif() -> str | None:
    """Animated GIF: one knee per KL grade with a caption – 'a knee through the grades'."""
    frames = []
    for g in range(NUM_CLASSES):
        files = split_images("train", g)
        if not files:
            return None
        img = cv2.cvtColor(cv2.resize(cv2.imread(str(files[3 % len(files)]), cv2.IMREAD_GRAYSCALE), (320, 320)), cv2.COLOR_GRAY2RGB)
        banner = np.full((56, 320, 3), 255, np.uint8)
        r, gg, b = (int(GRADE_COLOURS[g][i:i + 2], 16) for i in (1, 3, 5))
        banner[:, :] = (r, gg, b)
        cv2.putText(banner, f"{CLASS_NAMES[g]} - {KL_INFO[CLASS_NAMES[g]][0]}", (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (255, 255, 255), 2, cv2.LINE_AA)
        bar = np.full((14, 320, 3), 235, np.uint8)
        bar[:, : int(320 * (g + 1) / NUM_CLASSES)] = (r, gg, b)
        frames.append(Image.fromarray(np.vstack([banner, img, bar])))
    buf = io.BytesIO()
    frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:], duration=1300, loop=0)
    return base64.b64encode(buf.getvalue()).decode()


def show_gif(b64: str, width: int = 360) -> None:
    st.markdown(f"<div style='text-align:center;'><img src='data:image/gif;base64,{b64}' width='{width}' "
                f"style='border-radius:12px; box-shadow:0 4px 14px rgba(0,0,0,0.25);'></div>", unsafe_allow_html=True)


# ==============================================================================================
# Image preprocessing and models (identical pipeline to the notebook)
# ==============================================================================================
class RadiographPreprocessor:
    """Uploaded file → grayscale 224×224 → baseline or CLAHE."""

    def __init__(self, method: str = "baseline") -> None:
        self.method = method
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    @staticmethod
    def load(file_bytes: bytes) -> tuple[np.ndarray, dict]:
        img = Image.open(io.BytesIO(file_bytes))
        rgb = np.asarray(img.convert("RGB"))
        colour_spread = float(np.abs(rgb.astype(int) - rgb.mean(axis=2, keepdims=True)).mean())
        gray = np.asarray(img.convert("L"))
        h, w = gray.shape
        return gray, {"size": (w, h), "looks_colour": colour_spread > 8.0, "aspect": max(w, h) / max(1, min(w, h))}

    @staticmethod
    def resize(gray: np.ndarray) -> np.ndarray:
        return cv2.resize(gray, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)

    def process(self, gray_224: np.ndarray) -> np.ndarray:
        return self._clahe.apply(gray_224) if self.method == "clahe" else gray_224


class GradeModel(ABC):
    """A trained experiment: 224×224 grayscale radiograph → score per KL grade."""

    def __init__(self, exp_id: str, run: dict, model_file: Path) -> None:
        self.exp_id, self.run, self.model_file = exp_id, run, model_file
        self.family = run.get("family", "")
        self.preprocessor = RadiographPreprocessor(run["preprocessing"])

    @property
    def label(self) -> str:
        return f"{self.family} ({self.exp_id}, {self.run['imbalance']})"

    @abstractmethod
    def predict_proba(self, gray_224: np.ndarray) -> np.ndarray: ...

    def supports_heatmap(self) -> bool:
        return False


class ClassicalGradeModel(GradeModel):
    """HOG → StandardScaler → Linear SVM / Random Forest."""

    def __init__(self, exp_id, run, model_file) -> None:
        super().__init__(exp_id, run, model_file)
        m = re.match(r"hog_o(\d+)_c(\d+)_b(\d+)", run["signature"]["representation"])
        if m is None:
            raise ValueError("Prediction is only supported for HOG-based classical models.")
        self.orientations, self.cell, self.block = map(int, m.groups())
        saved = joblib.load(model_file)
        self.scaler, self.model = saved["scaler"], saved["model"]

    def predict_proba(self, gray_224):
        feats = hog(self.preprocessor.process(gray_224), orientations=self.orientations,
                    pixels_per_cell=(self.cell, self.cell), cells_per_block=(self.block, self.block),
                    block_norm="L2-Hys", feature_vector=True)
        x = self.scaler.transform(feats[None].astype(np.float32))
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(x)[0]
        scores = self.model.decision_function(x)[0]                     # SVM: softmax of decision scores
        e = np.exp(scores - scores.max())
        return e / e.sum()


class NeuralGradeModel(GradeModel):
    """Frozen ImageNet backbone → global average pooling → trained head (+ Grad-CAM)."""

    def __init__(self, exp_id, run, model_file) -> None:
        super().__init__(exp_id, run, model_file)
        import tensorflow as tf
        from tensorflow.keras import layers, models
        self.tf = tf
        constructor_name, module_name, weights_file = BACKBONES[self.family]
        constructor = getattr(tf.keras.applications, constructor_name)
        self.preprocess_input = getattr(tf.keras.applications, module_name).preprocess_input
        kwargs = dict(include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
        try:
            base = constructor(weights="imagenet", **kwargs)              # cached in ~/.keras by the notebook
        except Exception as err:
            local = WEIGHTS_DIR / weights_file
            if not local.exists():
                raise RuntimeError(f"ImageNet weights for {self.family} are not available offline.") from err
            base = constructor(weights=None, **kwargs)
            base.load_weights(str(local))
        base.trainable = False
        head = models.Sequential([layers.Input(shape=(base.output_shape[-1],)), layers.BatchNormalization(),
                                  layers.Dropout(0.2), layers.Dense(256, activation="relu"), layers.Dropout(0.5),
                                  layers.Dense(NUM_CLASSES, activation="softmax")], name="classifier_head")
        head.load_weights(str(model_file))
        feature_map = base.output
        probs = head(layers.GlobalAveragePooling2D()(feature_map), training=False)
        self.network = models.Model(base.input, [feature_map, probs])

    def _input(self, gray_224):
        x = np.repeat(self.preprocessor.process(gray_224)[None, ..., None].astype(np.float32), 3, axis=-1)
        return self.tf.constant(self.preprocess_input(x))

    def predict_proba(self, gray_224):
        return self.network(self._input(gray_224), training=False)[1].numpy()[0]

    def supports_heatmap(self) -> bool:
        return True

    def gradcam(self, gray_224: np.ndarray, class_index: int) -> np.ndarray:
        tf = self.tf
        with tf.GradientTape() as tape:
            fmap, probs = self.network(self._input(gray_224), training=False)
            score = probs[:, class_index]
        grads = tape.gradient(score, fmap)[0]
        cam = tf.nn.relu(tf.reduce_sum(fmap[0] * tf.reduce_mean(grads, axis=(0, 1)), axis=-1)).numpy()
        cam = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
        return cam / cam.max() if cam.max() > 0 else cam


@st.cache_resource(show_spinner="Loading model…")
def load_grade_model(exp_id: str) -> GradeModel:
    repo = get_repository()
    file = repo.model_file(exp_id)
    if file is None:
        raise FileNotFoundError(f"No saved model for {exp_id} in {RESULTS_DIR / exp_id}.")
    if file.suffix == ".joblib":
        return ClassicalGradeModel(exp_id, repo.runs[exp_id], file)
    if not TF_AVAILABLE:
        raise ModuleNotFoundError("TensorFlow is needed for the neural models.")
    return NeuralGradeModel(exp_id, repo.runs[exp_id], file)


def overlay_heatmap(gray_224: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    colour = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)[..., ::-1]
    return np.uint8((1 - alpha) * np.repeat(gray_224[..., None], 3, axis=-1) + alpha * colour)


def model_name(repo: ResultsRepository, exp_id: str) -> str:
    fam = repo.family(exp_id)
    star = " ⭐" if exp_id == repo.final_model() else ""
    return f"{MODEL_EMOJI.get(fam, '')} {fam} ({exp_id}){star}"


# ==============================================================================================
# Sidebar – user role
# ==============================================================================================
def sidebar_role() -> None:
    st.sidebar.title("User Role")
    st.sidebar.markdown(f"<div style='background:#ffffff; border-left:6px solid #2A9D8F; border-radius:10px; padding:10px 14px;'>"
                        f"<span style='font-size:1.6rem'>{ROLE_EMOJI}</span> <b>{USER_ROLE}</b><br>"
                        "<small>Built for doctors who grade knee X-rays on the Kellgren–Lawrence scale.</small></div>",
                        unsafe_allow_html=True)
    st.sidebar.caption("Tip: try the 🎮 *Challenge the AI* game on the Prediction page.")
    if not TF_AVAILABLE:
        st.sidebar.warning("Neural models off – TensorFlow not found (see Prediction page).")


# ==============================================================================================
# HOME
# ==============================================================================================
def page_home() -> None:
    set_bg(BACKGROUNDS["home"])
    repo = get_repository()
    page_header("Home")

    st.info("**Welcome, Radiologist!  \nYou grade these films every day – now meet an AI that tries to do the same.**  \n"
            "Compare your eye with the model's: where it is confident, where it hesitates (KL1 vs KL2), "
            "how often it is right for each grade, and what it actually looks at in the image.")
    st.write("Use the sidebar to navigate to the other pages: **Analysis** or **Prediction**.")

    gif = grade_journey_gif()
    if gif:
        show_gif(gif)
        st.caption("<div style='text-align:center'>A knee through the grades – watch the joint space narrow and the osteophytes grow "
                   "from KL0 to KL4 (real X-rays from the dataset).</div>", unsafe_allow_html=True)

    # ---- What does our dataset show? ------------------------------------------------------------------
    st.markdown("---")
    st.markdown("#### What does our dataset show?")
    counts = class_counts()
    if counts is None:
        st.error(f"No class counts found in `{RESULTS_DIR}`. Run the notebook first, then start the dashboard from the same folder.")
        return
    st.markdown("Use the selector below to see how the **8,260 knee X-rays** are spread across grades, data splits and diagnoses.")
    view = st.selectbox("Explore the X-rays by:", ["KL grade", "Data split", "Radiographic OA (KL ≥ 2) vs no OA"], key="home_dim")
    total = counts.sum(axis=0)
    if view == "KL grade":
        data = pd.DataFrame({"Group": CLASS_NAMES, "X-rays": total.values})
    elif view == "Data split":
        data = pd.DataFrame({"Group": ["Training", "Validation", "Test"], "X-rays": counts.sum(axis=1).values})
    else:
        data = pd.DataFrame({"Group": ["No OA (KL0–1)", "Radiographic OA (KL2–4)"],
                             "X-rays": [total[["KL0", "KL1"]].sum(), total[["KL2", "KL3", "KL4"]].sum()]})
    chart = alt.Chart(data).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
        x=alt.X("Group:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)), y=alt.Y("X-rays:Q"),
        color=alt.Color("Group:N", sort=None, scale=alt.Scale(scheme="oranges"), legend=None),
        tooltip=["Group", "X-rays"]).properties(height=380)
    labels = chart.mark_text(dy=-10, fontWeight="bold").encode(text="X-rays:Q", color=alt.value("#333"))
    st.altair_chart(chart + labels, width="stretch")

    kl0, kl4 = int(total["KL0"]), int(total["KL4"])
    with st.expander("Here's why our dataset is NOT balanced – and why that is a trap"):
        st.markdown(
            f"""
            Imagine a clinic where **40 out of every 100** knees are healthy (KL0) but only **3** are severe (KL4).
            In our data there are **{kl0:,} KL0** X-rays but only **{kl4:,} KL4** – about **{kl0 / kl4:.0f} healthy knees for every severe one**.

            1. A lazy "doctor" who says *"your knee is fine"* to **everyone** would already be right about **39%** of the time.
            2. An AI trained on this data learns the same shortcut – it is rewarded for saying KL0.
            3. The price is paid by the rare patients: severe knees get **missed**.

            👉 That is why this project used **imbalance-handling techniques** (class weighting, oversampling, augmentation)
            and judged models on **sensitivity per grade** and **macro-F1**, not on accuracy alone.
            """)
    with st.expander("What's found in the dataset"):
        st.markdown("### 📊 Key Insights")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("X-rays (train/val/test)", f"{int(total.sum()):,}")
        c2.metric("Most common grade", "KL0", f"{total['KL0'] / total.sum():.0%} of images", delta_color="off")
        c3.metric("Rarest grade", "KL4", f"{total['KL4'] / total.sum():.1%} of images", delta_color="off")
        c4.metric("Imbalance (train)", f"{counts.loc['train'].max() / counts.loc['train'].min():.1f} : 1")
        st.markdown("---")
        st.markdown("### 🧩 What this means for your reporting")
        st.markdown(
            "The data look like a real reporting list: lots of normal or doubtful knees, few severe ones. "
            "A model that is *'mostly right'* can still be **unsafe** if it misses the severe end. "
            "So when you look at the AI's answer, also look at **how often that model finds that grade** – "
            "the Prediction page tells you this for every result.")

    # ---- Learning quiz ------------------------------------------------------------------------------------
    final = repo.final_model() if repo.available else None
    with st.expander("🎓 Learning: Test your intuition"):
        st.info("Three quick questions about imbalanced data and the AI. No pressure – nobody is grading the grader here! 😉")
        quiz = [("1. A 'model' that answers **KL0 for every knee** would score what accuracy on our test set?",
                 ["5%", "20%", "39%", "75%"], "39%",
                 "39% of test knees really are KL0 – so **accuracy alone is misleading**: that 'model' misses every case of OA.")]
        if final:
            n4 = int(round(repo.runs[final]["test"]["kl4_recall"] * 51))
            quiz.append((f"2. Out of **51 severe (KL4) knees** in the test set, how many did the final model "
                         f"({repo.family(final)}) correctly call KL4?",
                         sorted({str(max(1, n4 // 3)), str(n4), str(min(51, n4 + 15)), "51"}, key=int), str(n4),
                         f"It found **{n4} of 51** ({n4 / 51:.0%}); most of the missed ones were called **KL3**, one grade away."))
            cm = np.asarray(repo.runs[final]["test"]["confusion_matrix"])
            pairs = {(i, j): cm[i, j] + cm[j, i] for i in range(NUM_CLASSES) for j in range(i + 1, NUM_CLASSES)}
            a, b = max(pairs, key=pairs.get)
            pair = f"{CLASS_NAMES[a]} ↔ {CLASS_NAMES[b]}"
            choices = sorted([pair] + [o for o in ["KL0 ↔ KL4", "KL1 ↔ KL3", "KL0 ↔ KL1", "KL1 ↔ KL2", "KL3 ↔ KL4"] if o != pair][:3])
            quiz.append(("3. Which pair of grades does the final AI model mix up **most often**?", choices, pair,
                         f"**{pair}** – {int(pairs[(a, b)])} test images were swapped between them. Neighbouring early grades "
                         "are the hardest, for radiologists and AI alike."))
        answers = [st.radio(q, opts, key=f"quiz_{k}", horizontal=True, index=None) for k, (q, opts, _, _) in enumerate(quiz)]
        if st.button("Check my answers"):
            score = sum(a == q[2] for a, q in zip(answers, quiz))
            for a, (q, _, right, why) in zip(answers, quiz):
                (st.success if a == right else st.warning)(("✅ " if a == right else f"⚠️ Answer: **{right}**. ") + why)
            st.markdown(f"### Your score: {score} / {len(quiz)}")
            if score == len(quiz):
                st.balloons()

    # ---- News / awareness ---------------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("#### News Report for Awareness")
    st.markdown("""
    **Osteoarthritis is one of the fastest-growing causes of disability worldwide.**

    The *Global Burden of Disease 2021* study (published in *The Lancet Rheumatology*, 2023) estimated that
    **595 million people** were living with osteoarthritis in **2020** – up **132%** from **256 million in 1990** –
    and projects that the number will **approach 1 billion by 2050**, with knee OA cases rising by about **75%**.
    """)
    left, right = st.columns([3, 2])
    with left:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=["1990", "2020"], y=[256, 595], name="Estimated", marker_color=["#F4A261", "#E76F51"],
                             text=["256 M", "595 M"], textposition="outside"))
        fig.add_trace(go.Bar(x=["2050 (projected)"], y=[1000], name="Projected", marker_color="#BDBDBD",
                             marker_pattern_shape="/", text=["≈1 billion"], textposition="outside"))
        fig.update_layout(title="People living with osteoarthritis (millions)", showlegend=False, height=360,
                          yaxis=dict(title="Millions of people", range=[0, 1150]), xaxis=dict(type="category"),
                          margin=dict(t=60, b=20))
        st.plotly_chart(fig, width="stretch")
    with right:
        fig = go.Figure(go.Pie(labels=["Women", "Men"], values=[61, 39], hole=0.55,
                               marker_colors=["#E76F51", "#2A9D8F"], textinfo="label+percent"))
        fig.update_layout(title="Who has OA? (2020)", showlegend=False, height=360, margin=dict(t=60, b=20),
                          annotations=[dict(text="61%<br>women", showarrow=False, font_size=16)])
        st.plotly_chart(fig, width="stretch")
    st.markdown("""
    According to the **World Health Organization**, about **528 million** people were living with osteoarthritis in 2019,
    and **the knee is the most frequently affected joint (365 million people)**. About **73%** are older than 55 and
    about **60%** are women.
    """)
    with st.expander("What helps patients? (WHO recommendations)"):
        st.markdown("""
        - **Exercise** to strengthen the muscles around the joint and keep mobility
        - **Healthy weight** – obesity accounted for about **20%** of OA-related disability in 2020 (GBD 2021)
        - **Education** and self-management, braces and assistive devices
        - **Pain relief** such as NSAIDs where appropriate
        - **Joint replacement** for severely affected joints
        - Most guidelines do **not** support opioids, glucosamine or visco-supplementation
        """)
    st.markdown("""
    **Sources:**
    [WHO – Osteoarthritis fact sheet (2023)](https://www.who.int/news-room/fact-sheets/detail/osteoarthritis) ·
    [GBD 2021 Osteoarthritis Collaborators, *Lancet Rheumatology* (2023)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10477960/) ·
    [IHME news release](https://www.eurekalert.org/news-releases/998866)
    """)

    # ---- Key takeaways ----------------------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("#### 💡 Key Takeaways")
    takeaways = [
        "Treat the AI as a **second reader**, not a replacement: its errors mostly fall **one grade away** "
        "(about 70% for the neural models) – just like disagreement between readers.",
        "**KL1 ('doubtful')** is the weakest grade for every model – the same grade on which human readers disagree most.",
        "Imbalance handling raised **KL4 sensitivity** – without it, one model found **none** of the severe knees. "
        "Most remaining misses are **KL4 → KL3**.",
        "A **'borderline'** result or a split vote between models is the cue to look harder, and the **Grad-CAM heat-map** "
        "shows whether the model looked at the joint space.",
        "Try the **🎮 Challenge the AI** game on the Prediction page and compare your grading with the model's.",
    ]
    st.markdown("\n".join(f"- {t}" for t in takeaways))
    samples = [split_images("test", g)[:1] for g in range(NUM_CLASSES)]
    if all(samples):
        cols = st.columns(NUM_CLASSES)
        for g, col in enumerate(cols):
            col.image(str(samples[g][0]), caption=f"{CLASS_NAMES[g]} · {KL_INFO[CLASS_NAMES[g]][0]}", width="stretch")
    st.markdown(
        "**❤️ Why this matters:**\n"
        "Knee osteoarthritis takes away walks, stairs, sleep and independence from hundreds of millions of people. "
        "Grading X-rays consistently is hard, and severe knees are easy to miss when they are rare. A tool that is honest "
        "about **what it sees, how sure it is, and how often it is right for each grade** can help doctors work faster "
        "and more consistently – while the final decision stays where it belongs: with the clinician and the patient.")
    st.caption(DISCLAIMER)


# ==============================================================================================
# ANALYSIS
# ==============================================================================================
def interpretation(text: str) -> None:
    with st.expander("**Interpretation**"):
        st.markdown(text)


def page_analysis() -> None:
    set_bg(BACKGROUNDS["analysis"])
    repo = get_repository()
    page_header("Analysis")
    st.markdown("---")
    if not repo.available:
        st.error(f"No results found in `{RESULTS_DIR}`. Run the notebook first.")
        return
    st.info("This section turns the experiment results into pictures a busy clinician can read in seconds.\n\n"
            "- See how rare severe knees are\n- Compare the four AI models side by side\n"
            "- Check whether fixing the imbalance really helped\n- Find out **where** the AI gets confused")
    best, final = repo.best_per_model(), repo.final_model()
    counts = class_counts()

    # ---- 1. Donut: grade distribution ---------------------------------------------------------------------
    section("1. How common is each KL grade?")
    split = st.radio("Show split:", ["train", "val", "test"], horizontal=True, key="donut_split",
                     format_func=lambda s: {"train": "Training", "val": "Validation", "test": "Test"}[s])
    row = counts.loc[split]
    fig = go.Figure(go.Pie(labels=[f"{g} · {KL_INFO[g][0]}" for g in CLASS_NAMES], values=row.values, hole=0.55,
                           marker_colors=GRADE_COLOURS, sort=False, direction="clockwise", textinfo="percent",
                           hovertemplate="%{label}<br>%{value} X-rays (%{percent})<extra></extra>"))
    fig.update_layout(height=420, annotations=[dict(text=f"KL4 = {row['KL4'] / row.sum():.1%}", showarrow=False, font_size=18)],
                      legend=dict(orientation="h", y=-0.1), margin=dict(t=20, b=20))
    st.plotly_chart(fig, width="stretch")
    interpretation(f"The slices are very unequal: KL0 fills **{row['KL0'] / row.sum():.0%}** of the {split} set while KL4 is only "
                   f"**{row['KL4'] / row.sum():.1%}**. The three splits look almost identical, so the test set reflects the same "
                   "real-world imbalance the model was trained on.")

    # ---- 2. Pictogram: the waiting room -------------------------------------------------------------------
    section("2. Your waiting room: who walks in?")
    n = st.slider("Number of patients in the waiting room", 20, 200, 100, 10, key="picto_n")
    share = counts.sum(axis=0) / counts.values.sum()
    raw = share.values * n
    k = np.floor(raw).astype(int)
    for idx in np.argsort(raw - k)[::-1][: n - k.sum()]:
        k[idx] += 1
    grades = np.repeat(np.arange(NUM_CLASSES), k)
    cols_n = 20
    xs, ys = np.arange(n) % cols_n, -(np.arange(n) // cols_n)
    fig = go.Figure()
    for g in range(NUM_CLASSES):
        mask = grades == g
        fig.add_trace(go.Scatter(x=xs[mask], y=ys[mask], mode="markers", name=f"{CLASS_NAMES[g]} ({k[g]})",
                                 marker=dict(symbol="circle", size=22 if n <= 100 else 15, color=GRADE_COLOURS[g],
                                             line=dict(color="white", width=1.5)),
                                 hovertemplate=f"{CLASS_NAMES[g]} – {KL_INFO[CLASS_NAMES[g]][0]}<extra></extra>"))
    fig.update_layout(height=120 + 32 * (n // cols_n + 1), xaxis=dict(visible=False), yaxis=dict(visible=False),
                      legend=dict(orientation="h", y=1.15), margin=dict(t=40, b=10), plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch")
    c1, c2, c3 = st.columns(3)
    c1.metric("🟢 Healthy (KL0)", f"{k[0]} patients")
    c2.metric("🟡 Early / doubtful (KL1–2)", f"{k[1] + k[2]} patients")
    c3.metric("🔴 Moderate–severe (KL3–4)", f"{k[3] + k[4]} patients", f"only {k[4]} severe", delta_color="off")
    interpretation(f"Each dot is one knee, coloured by grade, in the same proportions as the dataset. With **{n} patients**, "
                   f"only about **{k[4]}** have severe OA. If an AI learns that *'saying KL0 is usually safe'*, those few red dots "
                   "are exactly the ones it will miss.")

    # ---- 3. Radar: model comparison ------------------------------------------------------------------------
    section("3. Which AI model reads knees best?")
    chosen = st.multiselect("Models to compare", list(best), default=list(best), key="radar_models",
                            format_func=lambda m: f"{MODEL_EMOJI[m]} {m}")
    metrics = {"Macro-F1": "macro_f1", "Balanced accuracy": "balanced_accuracy", "KL4 sensitivity": "kl4_recall",
               "QWK (grade agreement)": "qwk", "Accuracy": "accuracy"}
    fig = go.Figure()
    palette = {"SVM": "#1f77b4", "Random Forest": "#2ca02c", "MobileNetV2": "#ff7f0e", "DenseNet121": "#d62728"}
    for m in chosen:
        t = repo.runs[best[m]]["test"]
        vals = [t[v] for v in metrics.values()]
        fig.add_trace(go.Scatterpolar(r=vals + vals[:1], theta=list(metrics) + list(metrics)[:1], fill="toself", opacity=0.55,
                                      name=f"{m} ({best[m]})", line_color=palette[m]))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 0.8], tickformat=".1f")), height=480,
                      legend=dict(orientation="h", y=-0.12), margin=dict(t=30, b=30))
    st.plotly_chart(fig, width="stretch")
    df = repo.summary(list(best.values()))
    leader = df.loc[df["Macro-F1"].idxmax()]
    st.success(f"🏆 **Final model (chosen on validation data): {repo.family(final)} ({final})**. "
               f"On the test set, the best balance across all five grades (macro-F1) came from **{leader['Model']} ({leader['Experiment']})** "
               f"with {leader['Macro-F1']:.3f}.")
    interpretation("A bigger shape is better. The two **neural networks** (orange, red) cover the most area: they agree best with "
                   "the true grade (QWK) and balance the grades best (macro-F1). The **SVM** is surprisingly strong at finding severe "
                   "knees (KL4 sensitivity), while the **Random Forest** is the weakest overall. Accuracy hardly differs between models – "
                   "another sign that accuracy alone doesn't tell you much here.")

    # ---- 4. Dumbbell: did fixing the imbalance help? --------------------------------------------------------
    section("4. Did fixing the imbalance help?")
    c1, c2 = st.columns(2)
    model = c1.selectbox("Model", MODEL_ORDER, index=MODEL_ORDER.index(repo.family(final)) if final else 0, key="db_model",
                         format_func=lambda m: f"{MODEL_EMOJI[m]} {m}")
    metric_label = c2.radio("Measure", ["KL4 sensitivity", "Macro-F1", "Accuracy"], horizontal=True, key="db_metric")
    summ = repo.summary([e for e in MAIN_EXPERIMENTS if e in repo.runs])
    pre = repo.runs[best[model]]["preprocessing"]
    sub = summ[(summ["Model"] == model) & (summ["Preprocessing"] == pre)].set_index("Imbalance technique")
    if "none" in sub.index:
        base_val = sub.loc["none", metric_label]
        fig = go.Figure()
        for tech in TECHNIQUES[1:]:
            if tech not in sub.index:
                continue
            v = sub.loc[tech, metric_label]
            colour = "#2E9E5B" if v >= base_val else "#D64545"
            fig.add_trace(go.Scatter(x=[base_val, v], y=[tech, tech], mode="lines", line=dict(color=colour, width=6), showlegend=False,
                                     hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=[base_val], y=[tech], mode="markers", marker=dict(size=16, color="#9E9E9E"),
                                     name="No handling", showlegend=tech == TECHNIQUES[1],
                                     hovertemplate=f"No handling: {base_val:.3f}<extra></extra>"))
            fig.add_trace(go.Scatter(x=[v], y=[tech], mode="markers+text", marker=dict(size=18, color=colour),
                                     text=[f"{v - base_val:+.3f}"], textposition="middle right", name="With technique",
                                     showlegend=tech == TECHNIQUES[1], hovertemplate=f"{tech}: {v:.3f}<extra></extra>"))
        fig.update_layout(height=330, xaxis=dict(title=metric_label, range=[0, max(0.75, sub[metric_label].max() + 0.1)]),
                          legend=dict(orientation="h", y=1.15), margin=dict(t=40, b=40))
        st.plotly_chart(fig, width="stretch")
        best_tech = sub.drop(index="none")[metric_label].idxmax()
        st.info(f"For **{model}**, the biggest gain in **{metric_label}** came from **{best_tech}** "
                f"({base_val:.3f} → {sub.loc[best_tech, metric_label]:.3f}).")
    interpretation("Grey dot = the model trained on the raw, imbalanced data; coloured dot = the same model with an imbalance "
                   "technique (green = better, red = worse). Switch the measure to **Accuracy** and you will often see a small drop – "
                   "the model says KL0 less often – while **KL4 sensitivity** and **macro-F1** go up. That trade-off is the whole point: "
                   "fewer missed severe knees in exchange for a few more cautious answers.")

    # ---- 5. Confusion heatmap + treemap of errors -----------------------------------------------------------
    section("5. Where does the AI get confused?")
    ids = [e for e in MAIN_EXPERIMENTS if e in repo.runs]
    exp_id = st.selectbox("Experiment", ids, index=ids.index(final), key="cm_exp",
                          format_func=lambda e: f"{e} – {repo.family(e)} | {repo.runs[e]['preprocessing']} | {repo.runs[e]['imbalance']}"
                          + (" ⭐" if e == final else ""))
    cm = np.asarray(repo.runs[exp_id]["test"]["confusion_matrix"])
    as_pct = st.toggle("Show as % of each true grade", value=True, key="cm_pct")
    z = cm / cm.sum(axis=1, keepdims=True) if as_pct else cm
    text = [[f"{v:.0%}" if as_pct else f"{int(v)}" for v in r] for r in z]
    left, right = st.columns([1.1, 1])
    with left:
        fig = go.Figure(go.Heatmap(z=z, x=CLASS_NAMES, y=CLASS_NAMES, text=text, texttemplate="%{text}", colorscale="Blues",
                                   showscale=False, hovertemplate="True %{y} → predicted %{x}: %{text}<extra></extra>"))
        fig.update_layout(height=420, xaxis_title="Predicted grade", yaxis_title="True grade",
                          yaxis=dict(autorange="reversed"), margin=dict(t=20, b=40))
        st.plotly_chart(fig, width="stretch")
    with right:
        err = [{"True grade": f"True {CLASS_NAMES[i]}", "Predicted as": f"→ {CLASS_NAMES[j]}", "Images": int(cm[i, j])}
               for i in range(NUM_CLASSES) for j in range(NUM_CLASSES) if i != j and cm[i, j] > 0]
        fig = px.treemap(pd.DataFrame(err), path=["True grade", "Predicted as"], values="Images", color="True grade",
                         color_discrete_sequence=GRADE_COLOURS)
        fig.update_layout(height=420, margin=dict(t=20, b=10, l=0, r=0))
        fig.update_traces(hovertemplate="%{label}: %{value} images<extra></extra>")
        st.plotly_chart(fig, width="stretch")
    off = cm.copy().astype(float)
    np.fill_diagonal(off, 0)
    i, j = np.unravel_index(off.argmax(), off.shape)
    dist = np.abs(np.subtract.outer(np.arange(NUM_CLASSES), np.arange(NUM_CLASSES)))
    adjacent = cm[dist == 1].sum() / max(1, cm[dist > 0].sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Correct on test", f"{np.trace(cm) / cm.sum():.1%}")
    c2.metric("Errors only one grade off", f"{adjacent:.0%}")
    c3.metric("Most common mix-up", f"{CLASS_NAMES[i]} → {CLASS_NAMES[j]}", f"{int(cm[i, j])} images", delta_color="off")
    interpretation("**Left:** each row is the true grade; the dark diagonal shows correct answers. **Right:** every box is a type of "
                   "mistake – the bigger the box, the more often it happens (click a box to zoom in). Most mistakes sit right next to "
                   "the diagonal: the AI rarely calls a severe knee healthy, but it often hesitates between **neighbouring grades**, "
                   "especially around KL1 – exactly where human readers disagree too.")

    # ---- 6. Report card per grade ---------------------------------------------------------------------------
    section("6. The AI's report card")
    model = st.selectbox("Model", list(best), index=list(best).index(repo.family(final)), key="rc_model",
                         format_func=lambda m: f"{MODEL_EMOJI[m]} {m} ({best[m]})")
    per = repo.runs[best[model]]["test"]["per_class"]

    def letter(v: float) -> str:
        return "A" if v >= 0.6 else "B" if v >= 0.45 else "C" if v >= 0.3 else "D"

    cols = st.columns(NUM_CLASSES)
    for g, col in zip(CLASS_NAMES, cols):
        rec = per[g]["recall"]
        col.markdown(f"<div class='kl-card' style='background:{GRADE_COLOURS[CLASS_NAMES.index(g)]}22; "
                     f"border-left:6px solid {GRADE_COLOURS[CLASS_NAMES.index(g)]}; text-align:center'>"
                     f"<b>{g}</b><br><span class='big-number'>{letter(rec)}</span><br>"
                     f"finds {rec:.0%}<br><small>right {per[g]['precision']:.0%} of the time when it says {g}</small></div>",
                     unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=CLASS_NAMES, y=[per[g]["recall"] for g in CLASS_NAMES], name="Sensitivity (finds the grade)",
                         marker_color="#2A9D8F", text=[f"{per[g]['recall']:.0%}" for g in CLASS_NAMES], textposition="outside"))
    fig.add_trace(go.Bar(x=CLASS_NAMES, y=[per[g]["precision"] for g in CLASS_NAMES], name="PPV (is right when it says it)",
                         marker_color="#E9C46A", text=[f"{per[g]['precision']:.0%}" for g in CLASS_NAMES], textposition="outside"))
    fig.update_layout(barmode="group", height=360, yaxis=dict(range=[0, 1.1], tickformat=".0%"),
                      legend=dict(orientation="h", y=1.15), margin=dict(t=40, b=20))
    st.plotly_chart(fig, width="stretch")
    interpretation("Grades: **A** ≥ 60% of that grade found, **B** ≥ 45%, **C** ≥ 30%, **D** below 30%. "
                   "**Sensitivity** answers *'if the knee is KL4, how often does the AI say KL4?'*; **PPV** answers "
                   "*'when the AI says KL4, how often is it right?'*. Expect the best marks at the two ends of the scale and the "
                   "lowest at **KL1** – the 'doubtful' grade.")

    # ---- 7. Learning curves -----------------------------------------------------------------------------------
    neural_ids = [e for e in ids if repo.runs[e].get("history", {}).get("loss")]
    if neural_ids:
        section("7. How the neural networks learned")
        exp_id = st.selectbox("Neural experiment", neural_ids, index=neural_ids.index(best.get("DenseNet121", neural_ids[0]))
                              if best.get("DenseNet121") in neural_ids else 0, key="curve_exp",
                              format_func=lambda e: f"{e} – {repo.family(e)} | {repo.runs[e]['imbalance']}")
        hist, best_ep = repo.runs[exp_id]["history"], repo.runs[exp_id].get("best_epoch")
        ep = list(range(1, len(hist["loss"]) + 1))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ep, y=hist["loss"], name="Training loss", line=dict(color="#264653", width=3)))
        fig.add_trace(go.Scatter(x=ep, y=hist["val_macro_f1"][:len(ep)], name="Validation macro-F1", yaxis="y2",
                                 line=dict(color="#E76F51", width=3)))
        if best_ep:
            fig.add_vline(x=best_ep, line_dash="dash", annotation_text=f"best epoch {best_ep} (kept)")
        fig.update_layout(height=380, hovermode="x unified", xaxis_title="Epoch",
                          yaxis=dict(title="Training loss"), yaxis2=dict(title="Validation macro-F1", overlaying="y", side="right",
                                                                         range=[0, 0.7]),
                          legend=dict(orientation="h", y=-0.25), margin=dict(t=30, b=30))
        st.plotly_chart(fig, width="stretch")
        interpretation("The training loss keeps falling – the network keeps 'memorising' – but the validation score stops improving "
                       "after a few epochs. Training was stopped automatically and the **best epoch** (dashed line) was kept, so the "
                       "model does not overfit to the training X-rays.")

    st.markdown("---")
    st.success("**Bottom line for radiologists:** the AI's errors look like human disagreement – mostly **one grade apart** – "
               "which makes it a useful consistency check and a safety net for severe knees, not a replacement reader. "
               "Give KL1 calls and KL4 → KL3 boundaries your closest look.")
    with st.expander("See every experiment (table)"):
        table = repo.summary()
        st.dataframe(table.round(3), hide_index=True, width="stretch")
        st.download_button("Download table (CSV)", table.to_csv(index=False).encode(), "knee_oa_experiments.csv", "text/csv")


# ==============================================================================================
# PREDICTION
# ==============================================================================================
def predict_all(repo: ResultsRepository, gray: np.ndarray, exp_id: str, heatmap: bool, compare: bool) -> dict:
    model = load_grade_model(exp_id)
    g224 = RadiographPreprocessor.resize(gray)
    probs = model.predict_proba(g224)
    pred_idx = int(np.argmax(probs))
    out = {"exp_id": exp_id, "gray": g224, "probs": probs, "pred_idx": pred_idx, "cam": None, "compare": []}
    if heatmap and model.supports_heatmap():
        out["cam"] = model.gradcam(g224, pred_idx)
    if compare:
        for other in repo.prediction_models():
            p = load_grade_model(other).predict_proba(g224)
            out["compare"].append({"Model": model_name(repo, other), "Technique": repo.runs[other]["imbalance"],
                                   "Predicted": CLASS_NAMES[int(np.argmax(p))],
                                   **{g: float(v) for g, v in zip(CLASS_NAMES, p)}})
    return out


def severity_gauge(probs: np.ndarray) -> go.Figure:
    expected = float(np.dot(probs, np.arange(NUM_CLASSES)))
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=expected, number=dict(valueformat=".1f", suffix=" / 4"),
        title=dict(text="Severity meter<br><span style='font-size:0.75em'>expected KL grade</span>"),
        gauge=dict(axis=dict(range=[0, 4], tickvals=[0, 1, 2, 3, 4], ticktext=CLASS_NAMES),
                   bar=dict(color="#263238", thickness=0.25),
                   steps=[dict(range=[max(0, g - 0.5), min(4, g + 0.5)], color=GRADE_COLOURS[g]) for g in range(NUM_CLASSES)])))
    fig.update_layout(height=280, margin=dict(t=70, b=10, l=30, r=30))
    return fig


def page_prediction() -> None:
    set_bg(BACKGROUNDS["prediction"])
    repo = get_repository()
    page_header("Prediction")
    st.markdown("---")
    st.warning(DISCLAIMER, icon="⚠️")
    if not repo.available:
        st.error(f"No results found in `{RESULTS_DIR}`. Run the notebook first.")
        return
    if not TF_AVAILABLE:
        tf_help()
    options = repo.prediction_models()
    if not options:
        st.error("No usable saved models found (results/<ID>/model.joblib or model.weights.h5).")
        return
    default = repo.default_model()
    has_test = (DATA_DIR / "test").is_dir()

    tab_pred, tab_game = st.tabs(["🩻 Predict a radiograph", "🎮 Challenge the AI"])

    # ---- Predict tab -------------------------------------------------------------------------------------------
    with tab_pred:
        st.info("**How to use:** choose an X-ray (upload your own or pick one from the test set), choose a model, then press "
                "**Predict KL grade**. Best results with a single-knee, front-view (AP) radiograph cropped around the joint.")
        source = st.radio("Radiograph source", ["Upload a radiograph", "Use a test-set example"], horizontal=True,
                          disabled=not has_test, key="pred_source")
        with st.form("predict_form", clear_on_submit=False):
            c1, c2 = st.columns(2)
            exp_id = c1.selectbox("AI model 🤖", options, index=options.index(default), format_func=lambda e: model_name(repo, e))
            with c2:
                heat = st.checkbox("Show Grad-CAM heat-map 🔥 (neural models)", value=True)
                compare = st.checkbox("Ask all models for a second opinion 👥", value=True)
            upload, grade, number = None, 4, 1
            if source == "Upload a radiograph":
                upload = st.file_uploader("Knee X-ray (PNG, JPG, JPEG or BMP)", type=["png", "jpg", "jpeg", "bmp"])
            else:
                c3, c4 = st.columns(2)
                grade = c3.selectbox("True grade of the example", range(NUM_CLASSES), index=4, format_func=lambda i: CLASS_NAMES[i])
                number = c4.number_input("Image number", min_value=1, max_value=999, value=1,
                                         help="If the number is larger than the images available, it wraps around.")
            submit = st.form_submit_button("🔍 Predict KL grade", type="primary")

        if submit:
            if source == "Upload a radiograph":
                if upload is None:
                    st.warning("Please upload an X-ray first.")
                    st.stop()
                file_bytes, name, true_grade = upload.getvalue(), upload.name, None
            else:
                files = split_images("test", grade)
                path = files[(int(number) - 1) % len(files)]
                file_bytes, name, true_grade = path.read_bytes(), path.name, CLASS_NAMES[grade]
            gray, checks = RadiographPreprocessor.load(file_bytes)
            try:
                with st.spinner("Reading the X-ray… 🩻"):
                    result = predict_all(repo, gray, exp_id, heat, compare)
            except Exception as err:                                       # never show a raw traceback to the user
                st.error(f"The model could not run: {err}")
                st.stop()
            result.update(name=name, true_grade=true_grade, checks=checks)
            st.session_state["prediction"] = result

        result = st.session_state.get("prediction")
        if result:
            show_prediction(repo, result)

    # ---- Game tab ----------------------------------------------------------------------------------------------
    with tab_game:
        challenge_game(repo, default, has_test)


def show_prediction(repo: ResultsRepository, r: dict) -> None:
    exp_id, probs, pred_idx = r["exp_id"], r["probs"], r["pred_idx"]
    pred = CLASS_NAMES[pred_idx]
    run = repo.runs[exp_id]
    st.markdown("---")
    if r["checks"]["looks_colour"]:
        st.warning("This image looks like a colour photo. The model expects a grayscale knee X-ray – treat the result with caution.")
    if r["checks"]["aspect"] > 1.3:
        st.warning(f"The image is far from square ({r['checks']['size'][0]}×{r['checks']['size'][1]}). The model was trained on "
                   "single-knee crops – crop around one knee joint for best results.")
    st.subheader(f"Predicted KL grade: **{pred} – {KL_INFO[pred][0]}**")
    c1, c2, c3 = st.columns([1, 1, 1.2])
    with c1:
        st.image(r["gray"], caption=f"{r['name']}" + (f" · reference grade {r['true_grade']}" if r["true_grade"] else ""),
                 width="stretch", clamp=True)
    with c2:
        if r["cam"] is not None:
            st.image(overlay_heatmap(r["gray"], r["cam"]), caption=f"🔥 Where the AI looked to decide '{pred}'", width="stretch")
        else:
            st.plotly_chart(severity_gauge(probs), width="stretch")
    with c3:
        st.markdown(f"<div class='kl-card' style='background:{GRADE_COLOURS[pred_idx]}22; border-left:8px solid {GRADE_COLOURS[pred_idx]}'>"
                    f"<div style='opacity:.8'>What {pred} means</div><div style='font-size:1.6rem;font-weight:800'>{pred} · "
                    f"{KL_INFO[pred][0]}</div>{KL_INFO[pred][1]}</div>", unsafe_allow_html=True)
        order = np.argsort(probs)[::-1]
        margin = probs[order[0]] - probs[order[1]]
        if margin < 0.15:
            st.info(f"🤔 **Borderline:** {pred} ({probs[order[0]]:.0%}) vs {CLASS_NAMES[order[1]]} ({probs[order[1]]:.0%}). "
                    "Consider both grades.")
        else:
            st.caption(f"Confidence gap to the next grade ({CLASS_NAMES[order[1]]}): {margin:.0%}")
        if r["true_grade"]:
            dist = abs(CLASS_NAMES.index(r["true_grade"]) - pred_idx)
            if dist == 0:
                st.success(f"🎯 Matches the reference grade ({r['true_grade']}).")
            else:
                st.error(f"Reference grade is **{r['true_grade']}** – the AI is {dist} grade{'s' if dist > 1 else ''} away.")
        pc = run["test"]["per_class"][pred]
        st.markdown(f"**How far can you trust this?** On 1,656 test X-rays this model found **{pc['recall']:.0%}** of true {pred} knees, "
                    f"and was right **{pc['precision']:.0%}** of the time when it answered {pred}.")

    c4, c5 = st.columns([1.4, 1])
    with c4:
        fig = go.Figure(go.Bar(x=CLASS_NAMES, y=probs, marker_color=GRADE_COLOURS, text=[f"{p:.0%}" for p in probs],
                               textposition="outside", hovertemplate="%{x}: %{y:.1%}<extra></extra>"))
        fig.update_layout(title="Score for every grade", height=320, yaxis=dict(range=[0, 1.1], tickformat=".0%"),
                          margin=dict(t=50, b=20))
        st.plotly_chart(fig, width="stretch")
    with c5:
        if r["cam"] is not None:
            st.plotly_chart(severity_gauge(probs), width="stretch")
        else:
            st.caption("Heat-maps are available for the neural models (MobileNetV2, DenseNet121)."
                       + ("" if TF_AVAILABLE else " Install TensorFlow to switch them on."))
    st.caption(("Scores are softmax-scaled SVM decision scores, not calibrated probabilities. "
                if run.get("family") == "SVM" else "Scores are model outputs, not calibrated clinical risk. ")
               + f"Model: {model_name(repo, exp_id)} – {run['imbalance']}.")

    kind, steps = NEXT_STEPS[pred]
    box = st.info if kind == "info" else st.warning
    box("💡 **Reporting notes** (general guidance, not a report):\n\n" + "\n".join(f"- {s}" for s in steps))

    if r["compare"]:
        st.markdown("#### 👥 Second opinions from all models")
        comp = pd.DataFrame(r["compare"])
        st.dataframe(comp.style.format({g: "{:.0%}" for g in CLASS_NAMES}).background_gradient(subset=CLASS_NAMES, cmap="Oranges", vmin=0, vmax=1),
                     hide_index=True, width="stretch")
        votes = comp["Predicted"].value_counts()
        if len(votes) == 1:
            st.success(f"✅ All models agree: **{votes.index[0]}**.")
        else:
            st.info("🗳️ The models disagree – " + ", ".join(f"**{g}** ({n} model{'s' if n > 1 else ''})" for g, n in votes.items())
                    + ". A split vote is a good reason to look at the film carefully.")

    lines = ["Knee OA Severity Dashboard – prediction summary", f"Image: {r['name']}", f"Model: {model_name(repo, exp_id)}",
             f"Estimated grade: {pred} ({KL_INFO[pred][0]})",
             "Scores: " + ", ".join(f"{g} {p:.3f}" for g, p in zip(CLASS_NAMES, probs))]
    if r["compare"]:
        lines.append("Second opinions: " + "; ".join(f"{c['Model']} {c['Predicted']}" for c in r["compare"]))
    lines.append(DISCLAIMER)
    st.download_button("📄 Download prediction summary (.txt)", "\n".join(lines).encode(), f"kl_prediction_{Path(r['name']).stem}.txt")


def challenge_game(repo: ResultsRepository, ai_exp: str, has_test: bool) -> None:
    """Doctor vs AI: guess the KL grade of a random test X-ray."""
    if not has_test:
        st.info("The game needs the `Knee_Dataset/test` folder next to this file.")
        return
    st.markdown(f"#### 🎮 Doctor vs AI\nGrade the knee yourself, then see what the AI ({model_name(repo, ai_exp)}) said and what the "
                "reference grade is. **Exact = 2 points, one grade off = 1 point.**")
    ss = st.session_state
    ss.setdefault("game", {"you": 0, "ai": 0, "rounds": 0, "path": None, "revealed": False, "history": []})
    game = ss["game"]

    def new_case() -> None:
        g = random.choices(range(NUM_CLASSES), weights=[1, 1, 1, 1, 1])[0]      # every grade equally likely – fairer game
        game["path"] = str(random.choice(split_images("test", g)))
        game["revealed"] = False

    if game["path"] is None:
        new_case()
    s1, s2, s3 = st.columns(3)
    s1.metric("👩‍⚕️ You", f"{game['you']} pts")
    s2.metric("🤖 AI", f"{game['ai']} pts")
    s3.metric("Rounds played", game["rounds"])

    left, right = st.columns([1, 1.3])
    path = Path(game["path"])
    true_idx = int(path.parent.name)
    with left:
        st.image(str(path), caption="Mystery knee", width=340)
    with right:
        guess = st.radio("Your grade:", CLASS_NAMES, horizontal=True, key="game_guess",
                         captions=[KL_INFO[g][0] for g in CLASS_NAMES], disabled=game["revealed"])
        b1, b2 = st.columns(2)
        if b1.button("🔒 Lock in my answer", disabled=game["revealed"], type="primary"):
            gray, _ = RadiographPreprocessor.load(path.read_bytes())
            try:
                p = load_grade_model(ai_exp).predict_proba(RadiographPreprocessor.resize(gray))
            except Exception as err:
                st.error(f"The AI could not run: {err}")
                return
            ai_idx, you_idx = int(np.argmax(p)), CLASS_NAMES.index(guess)
            pts = lambda i: 2 if i == true_idx else 1 if abs(i - true_idx) == 1 else 0
            game["you"] += pts(you_idx)
            game["ai"] += pts(ai_idx)
            game["rounds"] += 1
            game["revealed"] = True
            game["last"] = (you_idx, ai_idx, pts(you_idx), pts(ai_idx))
            game["history"].append({"Round": game["rounds"], "Reference": CLASS_NAMES[true_idx], "You": guess,
                                    "AI": CLASS_NAMES[ai_idx]})
            st.rerun()
        if b2.button("🎲 Next X-ray"):
            new_case()
            st.rerun()
        if game["revealed"]:
            you_idx, ai_idx, you_pts, ai_pts = game["last"]
            st.markdown(f"**Reference grade: {CLASS_NAMES[true_idx]} – {KL_INFO[CLASS_NAMES[true_idx]][0]}**  \n"
                        f"_{KL_INFO[CLASS_NAMES[true_idx]][1]}_")
            st.markdown(f"👩‍⚕️ You: **{CLASS_NAMES[you_idx]}** (+{you_pts}) · 🤖 AI: **{CLASS_NAMES[ai_idx]}** (+{ai_pts})")
            if you_pts > ai_pts:
                st.success("You beat the AI this round! 🏆")
            elif you_pts < ai_pts:
                st.warning("The AI wins this round 🤖 – press **Next X-ray** for a rematch.")
            else:
                st.info("It's a tie 🤝")
    if game["history"]:
        with st.expander("Game history"):
            st.dataframe(pd.DataFrame(game["history"]), hide_index=True, width="stretch")
        if st.button("Reset score"):
            ss["game"] = {"you": 0, "ai": 0, "rounds": 0, "path": None, "revealed": False, "history": []}
            st.rerun()


# ==============================================================================================
# Navigation
# ==============================================================================================
pages = [st.Page(page_home, title="Home", icon="🏠", default=True),
         st.Page(page_analysis, title="Analysis", icon="📊", url_path="analysis"),
         st.Page(page_prediction, title="Prediction", icon="🩻", url_path="prediction")]
nav = st.navigation(pages)
sidebar_role()
nav.run()
