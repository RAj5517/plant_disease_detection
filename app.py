import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as mpl_cm
from torchvision import models
from huggingface_hub import hf_hub_download
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
MODEL_REPO = "raj5517/plant-disease-resnet50"
MODEL_FILE = "best_model.pth"
DEVICE = torch.device("cpu")
NUM_CLASSES = 38

st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CSS  — dark botanical / scientific aesthetic
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

:root {
    --bg:        #0d1a0f;
    --surface:   #132016;
    --border:    #1e3322;
    --accent:    #4ade80;
    --accent2:   #86efac;
    --muted:     #4a7a55;
    --text:      #e8f5e9;
    --text-dim:  #7aab85;
    --warn-bg:   #2a1f00;
    --warn-bd:   #f59e0b;
    --ok-bg:     #0a2210;
    --ok-bd:     #4ade80;
}

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    color: var(--text);
}

.stApp {
    background: var(--bg) !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stSidebar"] .stSlider > div > div > div {
    background: var(--accent) !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: var(--surface);
    border: 1px dashed var(--border);
    border-radius: 12px;
}

/* Buttons */
.stButton > button {
    background: var(--accent) !important;
    color: #0d1a0f !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.06em !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.65rem 1.5rem !important;
    width: 100% !important;
    transition: background 0.2s !important;
}
.stButton > button:hover {
    background: var(--accent2) !important;
}

/* Divider */
hr { border-color: var(--border) !important; }

/* Captions */
[data-testid="stCaptionContainer"] p {
    color: var(--text-dim) !important;
    font-size: 0.78rem !important;
}

/* Radio + slider labels */
.stRadio label, .stSlider label {
    color: var(--text-dim) !important;
    font-size: 0.82rem !important;
}

/* Hero */
.hero-wrap {
    padding: 0.5rem 0 1.5rem 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.8rem;
}
.hero-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.18em;
    color: var(--accent);
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    color: var(--text);
    line-height: 1.1;
    margin-bottom: 0.5rem;
}
.hero-sub {
    font-size: 0.95rem;
    color: var(--text-dim);
    font-weight: 300;
    max-width: 520px;
}

/* Stat chips */
.stat-row {
    display: flex;
    gap: 0.6rem;
    margin-top: 1rem;
    flex-wrap: wrap;
}
.stat-chip {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 99px;
    padding: 0.3rem 0.75rem;
    color: var(--accent2);
    white-space: nowrap;
}

/* Result card */
.result-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
}
.result-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.4rem;
}
.result-value {
    font-family: 'Playfair Display', serif;
    font-size: 1.4rem;
    color: var(--text);
    line-height: 1.3;
}
.result-conf {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    color: var(--accent);
    margin-top: 0.3rem;
}

/* Banners */
.banner {
    border-radius: 10px;
    padding: 0.7rem 1rem;
    font-size: 0.88rem;
    font-weight: 500;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.banner-ok  { background: var(--ok-bg);   border: 1px solid var(--ok-bd);   color: #86efac; }
.banner-warn{ background: var(--warn-bg); border: 1px solid var(--warn-bd); color: #fcd34d; }

/* Top-N bars */
.bar-section { margin-top: 0.2rem; }
.bar-row { margin-bottom: 0.9rem; }
.bar-meta {
    display: flex;
    justify-content: space-between;
    font-size: 0.82rem;
    margin-bottom: 0.3rem;
}
.bar-name { color: var(--text); font-weight: 400; }
.bar-pct  { font-family: 'IBM Plex Mono', monospace; font-weight: 500; }
.bar-track {
    background: var(--border);
    border-radius: 99px;
    height: 5px;
    overflow: hidden;
}
.bar-fill { height: 100%; border-radius: 99px; }

/* Section header */
.section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.8rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid var(--border);
}

/* Placeholder box */
.placeholder {
    background: var(--surface);
    border: 1px dashed var(--border);
    border-radius: 14px;
    padding: 4rem 2rem;
    text-align: center;
    color: var(--muted);
    font-size: 0.88rem;
    margin-top: 2rem;
}
.placeholder-icon { font-size: 2rem; margin-bottom: 0.8rem; }

/* Footer */
.footer {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: var(--muted);
    text-align: center;
    padding: 1.5rem 0 0.5rem;
    border-top: 1px solid var(--border);
    margin-top: 2rem;
    letter-spacing: 0.05em;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# CLASS NAMES
# ─────────────────────────────────────────────────────────────
with open("class_names.txt") as f:
    CLASS_NAMES = [line.strip() for line in f if line.strip()]


# ─────────────────────────────────────────────────────────────
# MODEL  — matches training: build_resnet50(), bare state dict
# ─────────────────────────────────────────────────────────────
def build_resnet50(num_classes: int = NUM_CLASSES) -> nn.Module:
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, num_classes),
    )
    return model


@st.cache_resource(show_spinner="Loading model from Hugging Face Hub...")
def load_model() -> nn.Module:
    model = build_resnet50()
    path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
    # Saved as bare state dict (no checkpoint wrapper)
    state_dict = torch.load(path, map_location=DEVICE, weights_only=False)
    # Handle both bare state dict AND checkpoint dict (future-proof)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


model = load_model()


# ─────────────────────────────────────────────────────────────
# TRANSFORM  — matches training val/test pipeline
# ─────────────────────────────────────────────────────────────
inference_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


# ─────────────────────────────────────────────────────────────
# GRAD-CAM
# ─────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model: nn.Module):
        self.model = model
        self._features = None
        self._grads = None
        target = model.layer4[-1]
        target.register_forward_hook(lambda m, i, o: setattr(self, "_features", o))
        target.register_full_backward_hook(lambda m, gi, go: setattr(self, "_grads", go[0]))

    def generate(self, x: torch.Tensor, cls: int) -> np.ndarray:
        self.model.zero_grad()
        out = self.model(x)
        out[0, cls].backward()
        w = self._grads.mean(dim=(2, 3), keepdim=True)
        cam = (w * self._features).sum(dim=1).squeeze()
        cam = torch.clamp(cam, min=0).detach().cpu().numpy()
        cam = cv2.resize(cam, (224, 224))
        lo, hi = cam.min(), cam.max()
        return (cam - lo) / (hi - lo + 1e-8)


@st.cache_resource
def get_gradcam() -> GradCAM:
    return GradCAM(model)


gradcam = get_gradcam()


def cam_to_rgb(cam: np.ndarray) -> np.ndarray:
    return (mpl_cm.get_cmap("plasma")(cam)[:, :, :3] * 255).astype(np.uint8)


def blend_cam(img: np.ndarray, cam: np.ndarray, alpha: float = 0.50) -> np.ndarray:
    img224 = cv2.resize(img, (224, 224))
    heat = cam_to_rgb(cam)
    return np.clip(alpha * heat + (1 - alpha) * img224, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def fmt(raw: str) -> str:
    """'Tomato___Early_blight' → 'Tomato / Early Blight'"""
    parts = raw.replace("___", "|||").split("|||")
    if len(parts) == 2:
        return f"{parts[0].replace('_',' ').strip()} / {parts[1].replace('_',' ').strip().title()}"
    return raw.replace("_", " ").title()


def bar_color(p: float) -> str:
    if p >= 0.85: return "#4ade80"
    if p >= 0.60: return "#fbbf24"
    return "#f87171"


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("#### Settings")

    threshold = st.slider(
        "Confidence threshold",
        min_value=0.30, max_value=0.99, value=0.70, step=0.01,
        help="Predictions below this value trigger a low-confidence warning.",
    )
    st.caption(f"Warning shown if top confidence < {threshold*100:.0f}%")

    st.divider()

    cam_mode = st.radio(
        "Grad-CAM view",
        ["Overlay", "Side-by-side"],
        help="Overlay blends heatmap onto the image. Side-by-side shows both separately.",
    )

    st.divider()

    top_n = st.slider("Top-N predictions", min_value=1, max_value=5, value=3)

    st.divider()
    st.markdown(
        "<small style='color:#4a7a55;font-family:monospace;'>"
        "Model — ResNet50<br>"
        "Data — PlantVillage<br>"
        "Classes — 38<br>"
        "Test acc — 99.20%</small>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
    <div class="hero-eyebrow">Deep Learning · Plant Pathology</div>
    <div class="hero-title">Plant Disease<br>Detector</div>
    <div class="hero-sub">
        Upload a leaf photograph to identify plant disease and visualise
        model attention using Grad-CAM.
    </div>
    <div class="stat-row">
        <span class="stat-chip">ResNet50</span>
        <span class="stat-chip">38 classes</span>
        <span class="stat-chip">99.20% test acc</span>
        <span class="stat-chip">Macro F1 0.9901</span>
        <span class="stat-chip">PlantVillage · 54,305 imgs</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────────────────────────
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown('<div class="section-header">Input</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Drop a leaf image here",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, use_column_width=True)

with right:
    if not uploaded:
        st.markdown("""
        <div class="placeholder">
            <div class="placeholder-icon">🍃</div>
            Upload a leaf image on the left to begin analysis.
        </div>
        """, unsafe_allow_html=True)
    else:
        if st.button("Run Analysis"):

            img_np = np.array(image)
            x = inference_transform(image=img_np)["image"].unsqueeze(0).to(DEVICE)

            # ── Inference ──────────────────────────────
            with torch.no_grad():
                probs = F.softmax(model(x), dim=1)

            top_probs, top_idxs = torch.topk(probs, top_n)
            top_cls   = top_idxs[0][0].item()
            top_conf  = top_probs[0][0].item()
            top_label = CLASS_NAMES[top_cls]

            # ── Grad-CAM ───────────────────────────────
            with torch.enable_grad():
                cam_map = gradcam.generate(x.clone().requires_grad_(True), top_cls)

            # ── Confidence banner ──────────────────────
            if top_conf >= threshold:
                st.markdown(
                    f'<div class="banner banner-ok">'
                    f'<span>✓</span>'
                    f'<span>Confident — <strong>{top_conf*100:.1f}%</strong> '
                    f'(threshold {threshold*100:.0f}%)</span></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="banner banner-warn">'
                    f'<span>⚠</span>'
                    f'<span>Low confidence — <strong>{top_conf*100:.1f}%</strong> '
                    f'below {threshold*100:.0f}% threshold. '
                    f'Try a sharper or better-lit image.</span></div>',
                    unsafe_allow_html=True,
                )

            # ── Top-1 result card ──────────────────────
            st.markdown(
                f"""<div class="result-card">
                    <div class="result-label">Top Prediction</div>
                    <div class="result-value">{fmt(top_label)}</div>
                    <div class="result-conf">{top_conf*100:.2f}% confidence</div>
                </div>""",
                unsafe_allow_html=True,
            )

            # ── Top-N bars ─────────────────────────────
            st.markdown(
                f'<div class="section-header">Top {top_n} Predictions</div>',
                unsafe_allow_html=True,
            )
            bars = ""
            for i in range(top_n):
                cls   = CLASS_NAMES[top_idxs[0][i].item()]
                prob  = top_probs[0][i].item()
                color = bar_color(prob)
                bars += f"""
                <div class="bar-row">
                    <div class="bar-meta">
                        <span class="bar-name">{fmt(cls)}</span>
                        <span class="bar-pct" style="color:{color};">{prob*100:.1f}%</span>
                    </div>
                    <div class="bar-track">
                        <div class="bar-fill" style="width:{prob*100:.1f}%;background:{color};"></div>
                    </div>
                </div>"""
            st.markdown(f'<div class="bar-section">{bars}</div>', unsafe_allow_html=True)

            # ── Grad-CAM ───────────────────────────────
            st.markdown(
                '<div class="section-header" style="margin-top:1.2rem;">Grad-CAM · Model Attention</div>',
                unsafe_allow_html=True,
            )

            if cam_mode == "Overlay":
                st.image(
                    blend_cam(img_np, cam_map),
                    use_column_width=True,
                    caption="Heatmap overlay — bright regions = highest model attention",
                )
            else:
                c1, c2 = st.columns(2)
                with c1:
                    st.image(
                        cv2.resize(img_np, (224, 224)),
                        use_column_width=True,
                        caption="Original (224×224)",
                    )
                with c2:
                    st.image(
                        cam_to_rgb(cam_map),
                        use_column_width=True,
                        caption="Grad-CAM heatmap",
                    )

            st.caption(
                "Grad-CAM hooks into layer4 of ResNet50. "
                "Bright (yellow/white) regions drove the prediction most strongly."
            )


# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown(
    "<div class='footer'>"
    "ResNet50 &nbsp;·&nbsp; PlantVillage (54,305 images · 38 classes) &nbsp;·&nbsp; "
    "Test Acc 99.20% &nbsp;·&nbsp; Macro F1 0.9901 &nbsp;·&nbsp; "
    "Two-phase fine-tuning &nbsp;·&nbsp; "
    "Model hosted on Hugging Face Hub"
    "</div>",
    unsafe_allow_html=True,
)