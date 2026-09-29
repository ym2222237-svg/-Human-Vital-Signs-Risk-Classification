
import os
import io
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Optional ML / DL imports
import joblib
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.transforms import functional as TF

st.set_page_config(
    page_title="Hospital AI Screening",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# CONFIG
# =========================================================
VITAL_MODEL_PATH = "models/vital_signs_model.joblib"
BREAST_MODEL_PATH = "models/breast_cancer_model.pt"

VITAL_FEATURES = [
    "Heart_Rate",
    "Respiratory_Rate",
    "SpO2",
    "Temperature",
]

VITAL_LABELS = {
    0: "Low Risk",
    1: "High Risk",
}

# =========================================================
# THEME / CSS
# =========================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --navy: #0B2138;
        --navy-2: #132C46;
        --blue: #2E7BB8;
        --teal: #14A79C;
        --red: #D64550;
        --green: #2FA968;
        --amber: #C98A1F;
        --bg: #EEF3F8;
        --surface: #FFFFFF;
        --border: #E1E8F0;
        --muted: #62748A;
        --shadow-sm: 0 2px 10px rgba(11,33,56,.05);
        --shadow-md: 0 10px 28px rgba(11,33,56,.08);
        --shadow-lg: 0 20px 48px rgba(11,33,56,.14);
    }

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', 'Manrope', -apple-system, sans-serif;
    }

    .stApp {
        background:
            radial-gradient(1100px 480px at 15% -8%, rgba(20,167,156,.08), transparent 60%),
            radial-gradient(900px 420px at 100% 0%, rgba(46,123,184,.10), transparent 55%),
            var(--bg);
    }

    .block-container {
        max-width: 1360px;
        padding-top: 1.4rem;
        padding-bottom: 3rem;
    }

    /* ---------- Hero ---------- */
    .hero {
        position: relative;
        overflow: hidden;
        background: linear-gradient(120deg, #0B2138 0%, #16385A 48%, #0F8478 100%);
        padding: 36px 40px;
        border-radius: 24px;
        color: white;
        margin-bottom: 26px;
        box-shadow: var(--shadow-lg);
        border: 1px solid rgba(255,255,255,.06);
    }

    .hero::before {
        content: "";
        position: absolute;
        top: -60%;
        right: -8%;
        width: 420px;
        height: 420px;
        background: radial-gradient(circle, rgba(255,255,255,.10) 0%, transparent 70%);
        pointer-events: none;
    }

    .hero-eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(255,255,255,.12);
        border: 1px solid rgba(255,255,255,.18);
        color: #E7F3FF;
        font-size: .74rem;
        font-weight: 700;
        letter-spacing: .06em;
        text-transform: uppercase;
        padding: 6px 14px;
        border-radius: 999px;
        margin-bottom: 14px;
    }

    .hero h1 {
        margin: 0;
        font-size: 2.35rem;
        font-weight: 800;
        letter-spacing: -.01em;
        position: relative;
    }

    .hero p {
        margin: 10px 0 0;
        color: #D8E9F2;
        font-size: 1.02rem;
        max-width: 640px;
        position: relative;
    }

    /* ---------- Sections ---------- */
    .section-title {
        color: var(--navy);
        font-size: 1.4rem;
        font-weight: 800;
        letter-spacing: -.01em;
        margin: 6px 0 14px;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 22px 24px;
        box-shadow: var(--shadow-sm);
        line-height: 1.55;
    }

    .metric-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 18px;
        text-align: center;
        transition: box-shadow .15s ease, transform .15s ease;
        box-shadow: var(--shadow-sm);
    }

    .metric-card:hover {
        box-shadow: var(--shadow-md);
        transform: translateY(-1px);
    }

    .metric-label {
        color: var(--muted);
        font-size: .78rem;
        font-weight: 700;
        letter-spacing: .03em;
        text-transform: uppercase;
    }

    .metric-value {
        color: var(--navy);
        font-size: 1.65rem;
        font-weight: 800;
        margin-top: 6px;
    }

    /* ---------- Status banner ---------- */
    .status {
        border-radius: 16px;
        padding: 20px 24px;
        color: white;
        font-weight: 800;
        font-size: 1.25rem;
        margin: 14px 0;
        box-shadow: var(--shadow-md);
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .status-normal   { background: linear-gradient(120deg, #2FA968, #1F8A54); }
    .status-abnormal { background: linear-gradient(120deg, #E15561, #C6303C); }
    .status-neutral  { background: linear-gradient(120deg, #16385A, #0B2138); }
    .status-cancer   { background: linear-gradient(120deg, #E15561, #C6303C); }
    .status-no-cancer{ background: linear-gradient(120deg, #2FA968, #1F8A54); }

    /* ---------- Disclaimer ---------- */
    .disclaimer {
        background: #FFF8EA;
        border: 1px solid #F0D39A;
        color: #6B4E14;
        padding: 16px 18px;
        border-radius: 14px;
        font-size: .88rem;
        line-height: 1.5;
        margin-top: 18px;
    }

    .small-muted {
        color: var(--muted);
        font-size: .86rem;
    }

    /* ---------- Buttons ---------- */
    .stButton > button {
        border-radius: 12px;
        font-weight: 700;
        min-height: 46px;
        border: none;
        transition: transform .12s ease, box-shadow .12s ease;
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(120deg, #14A79C, #2E7BB8);
        box-shadow: 0 8px 20px rgba(20,167,156,.28);
    }

    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 10px 26px rgba(20,167,156,.36);
    }

    /* ---------- Tabs ---------- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: var(--surface);
        padding: 6px;
        border-radius: 14px;
        border: 1px solid var(--border);
        box-shadow: var(--shadow-sm);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        font-weight: 700;
        color: var(--muted);
        padding: 10px 18px;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(120deg, #0B2138, #16385A) !important;
        color: white !important;
    }

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0B2138 0%, #12314E 100%);
        border-right: 1px solid rgba(255,255,255,.06);
    }

    [data-testid="stSidebar"] * {
        color: #E7EEF5 !important;
    }

    [data-testid="stSidebar"] .stCaption, [data-testid="stSidebar"] small {
        color: #9FB4C7 !important;
    }

    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,.10);
    }

    [data-testid="stSidebar"] .stButton > button {
        background: rgba(255,255,255,.08);
        border: 1px solid rgba(255,255,255,.16) !important;
        color: #fff !important;
    }

    /* ---------- Login ---------- */
    .login-wrap {
        max-width: 460px;
        margin: 9vh auto 0 auto;
    }

    .login-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 24px;
        padding: 40px 36px 30px;
        box-shadow: var(--shadow-lg);
    }

    .login-badge {
        width: 62px;
        height: 62px;
        margin: 0 auto 18px;
        border-radius: 16px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.8rem;
        background: linear-gradient(135deg, #14A79C, #2E7BB8);
        box-shadow: 0 10px 22px rgba(20,167,156,.32);
    }

    .login-brand {
        text-align: center;
        color: var(--navy);
        font-size: 1.55rem;
        font-weight: 800;
        margin-bottom: 4px;
        letter-spacing: -.01em;
    }

    .login-subtitle {
        text-align: center;
        color: var(--muted);
        margin-bottom: 26px;
        font-size: .92rem;
    }

    .login-note {
        text-align: center;
        color: var(--muted);
        font-size: .8rem;
        margin-top: 16px;
    }

    [data-testid="stForm"] {
        border: none;
        padding: 0;
    }

    .stTextInput > div > div input {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# AUTHENTICATION — PRD REQUIREMENTS FR-1.1 to FR-1.5
# =========================================================
import hashlib
import base64
import hmac

AUTH_USERNAME = "admin"
AUTH_SALT_B64 = "aG9zcGl0YWwtYWktZGVtby1zYWx0LTIwMjY="
AUTH_PASSWORD_HASH_B64 = "2ytpSsaUDu4riDCKlEJptGnF49Fg+p3xNymUBFx5PWk="
AUTH_ITERATIONS = 200000


def _password_hash(password: str) -> str:
    salt = base64.b64decode(AUTH_SALT_B64)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        AUTH_ITERATIONS,
    )
    return base64.b64encode(digest).decode("utf-8")


def authenticate(username: str, password: str) -> bool:
    if username.strip() != AUTH_USERNAME:
        return False
    candidate = _password_hash(password)
    return hmac.compare_digest(candidate, AUTH_PASSWORD_HASH_B64)


if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.markdown(
        """
        <div class="login-wrap">
            <div class="login-card">
                <div class="login-badge">🏥</div>
                <div class="login-brand">Hospital Multi-Model AI</div>
                <div class="login-subtitle">Secure access to the AI screening dashboard</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _, login_col, _ = st.columns([1, 2, 1])
    with login_col:
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Username", value="admin", placeholder="Enter username")
            password = st.text_input(
                "Password",
                value="Hospital@123",
                type="password",
                placeholder="Enter password",
            )
            login = st.form_submit_button(
                "Login",
                type="primary",
                use_container_width=True,
            )

        if login:
            if authenticate(username, password):
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Invalid username or password.")

        st.markdown(
            '<div class="login-note">Authentication is required before accessing the dashboard.</div>',
            unsafe_allow_html=True,
        )

    st.stop()

# =========================================================
# ENVIRONMENT CHECK
# =========================================================
try:
    import sklearn
    SKLEARN_VERSION = sklearn.__version__
except Exception:
    SKLEARN_VERSION = "unknown"

# The Vital-Signs joblib was produced with a compatible sklearn version.
# Keep this environment pinned to sklearn 1.6.1.
if SKLEARN_VERSION != "1.6.1":
    st.warning(
        f"scikit-learn {SKLEARN_VERSION} is installed. "
        "The Vital-Signs model is expected to run with scikit-learn 1.6.1. "
        "Install the pinned requirements in a clean virtual environment."
    )

# =========================================================
# HELPERS
# =========================================================
def card_metric(label, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def load_vital_model():
    """Load the trained vital-signs sklearn model/pipeline safely."""
    if not os.path.exists(VITAL_MODEL_PATH):
        return None

    try:
        obj = joblib.load(VITAL_MODEL_PATH)

        # Support either a Pipeline itself or a checkpoint dictionary.
        if isinstance(obj, dict):
            for key in ("model", "pipeline", "best_model", "estimator"):
                if key in obj:
                    obj = obj[key]
                    break
            else:
                raise ValueError(
                    "The vital model file is a dictionary, but no model key was found. "
                    f"Available keys: {list(obj.keys())}"
                )

        if not hasattr(obj, "predict"):
            raise ValueError(
                f"Loaded vital model does not have predict(). Type: {type(obj)}"
            )

        return obj

    except Exception as e:
        st.error(f"Could not load vital-sign model: {e}")
        return None

class ResizeWithPad:
    def __init__(self, size, fill=0):
        self.size = size
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        scale = self.size / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h))
        pad_left = (self.size - new_w) // 2
        pad_top = (self.size - new_h) // 2
        pad_right = self.size - new_w - pad_left
        pad_bottom = self.size - new_h - pad_top
        return TF.pad(
            img,
            [pad_left, pad_top, pad_right, pad_bottom],
            fill=self.fill,
        )

class EfficientNetMultiTask(nn.Module):
    """EfficientNet-B0 image-only multi-task model.

    The model predicts Cancer, View, and Density directly from the image.
    View and Density are auxiliary image-classification heads and therefore
    do not require user selection at inference time.
    """

    def __init__(self, num_views, num_densities):
        super().__init__()

        self.backbone = models.efficientnet_b0(weights=None)
        image_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

        self.cancer_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(image_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

        self.view_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(image_features, 128),
            nn.ReLU(),
            nn.Linear(128, num_views),
        )

        self.density_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(image_features, 128),
            nn.ReLU(),
            nn.Linear(128, num_densities),
        )

    def forward(self, images):
        image_features = self.backbone(images)

        cancer_logits = self.cancer_head(image_features)
        view_logits = self.view_head(image_features)
        density_logits = self.density_head(image_features)

        return cancer_logits, view_logits, density_logits


@st.cache_resource
def load_breast_model():
    if not os.path.exists(BREAST_MODEL_PATH):
        return None, None

    try:
        checkpoint = torch.load(
            BREAST_MODEL_PATH,
            map_location="cpu",
        )

        state_dict = checkpoint.get("model_state_dict")
        if state_dict is None:
            raise ValueError(
                "The breast model checkpoint does not contain "
                "'model_state_dict'."
            )

        # The notebook model is the image-only multi-task architecture:
        # Cancer + View + Density.
        required_heads = (
            "cancer_head.",
            "view_head.",
            "density_head.",
        )

        if not all(
            any(key.startswith(prefix) for key in state_dict.keys())
            for prefix in required_heads
        ):
            raise ValueError(
                "The loaded checkpoint is not the multi-task model. "
                "The automatic View/Density feature requires a checkpoint "
                "trained with cancer_head, view_head, and density_head."
            )

        img_size = 512
        threshold = 0.05

        view_classes = checkpoint.get(
            "view_classes",
            ["AT", "CC", "LM", "LMO", "ML", "MLO", "Missing"],
        )
        density_classes = checkpoint.get(
            "density_classes",
            ["A", "B", "C", "D", "Missing"],
        )

        if isinstance(view_classes, dict):
            view_classes = [
                value
                for _, value in sorted(
                    view_classes.items(),
                    key=lambda x: int(x[0]),
                )
            ]

        if isinstance(density_classes, dict):
            density_classes = [
                value
                for _, value in sorted(
                    density_classes.items(),
                    key=lambda x: int(x[0]),
                )
            ]

        view_classes = list(view_classes)
        density_classes = list(density_classes)

        model = EfficientNetMultiTask(
            num_views=len(view_classes),
            num_densities=len(density_classes),
        )

        model.load_state_dict(state_dict, strict=True)
        model.eval()

        return model, {
            "img_size": img_size,
            "threshold": threshold,
            "class_names": checkpoint.get(
                "class_names",
                {0: "No Cancer", 1: "Cancer"},
            ),
            "view_classes": view_classes,
            "density_classes": density_classes,
            "view_to_idx": {
                value: idx for idx, value in enumerate(view_classes)
            },
            "density_to_idx": {
                value: idx for idx, value in enumerate(density_classes)
            },
            "automatic_metadata": True,
        }

    except Exception as e:
        st.error(f"Could not load breast-cancer model: {e}")
        return None, None


def breast_transform(img_size):
    return transforms.Compose([
        ResizeWithPad(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

def vital_explanation(model, values):
    """
    Explain the prediction without claiming causal medical meaning.
    Works with:
      - sklearn Pipeline whose final estimator exposes coef_
      - estimator exposing feature_importances_
    """
    estimator = model
    if hasattr(model, "named_steps"):
        estimator = list(model.named_steps.values())[-1]

    importance = None

    if hasattr(estimator, "feature_importances_"):
        importance = np.asarray(estimator.feature_importances_).reshape(-1)
    elif hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_)
        importance = np.abs(coef[0]).reshape(-1)

    if importance is None or len(importance) != len(VITAL_FEATURES):
        return None

    df = pd.DataFrame({
        "Feature": VITAL_FEATURES,
        "Importance": importance,
        "Value": values,
    }).sort_values("Importance", ascending=False)

    return df

def gradcam(
    model,
    input_tensor,
    target_layer,
):
    """
    Lightweight Grad-CAM for the image-only multi-task model.
    Grad-CAM targets the cancer logit only.
    """
    activations = []
    gradients = []

    def forward_hook(module, inp, out):
        activations.append(out)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

    try:
        model.zero_grad(set_to_none=True)

        cancer_logits, _, _ = model(input_tensor)
        logit = cancer_logits.squeeze()

        logit.backward()

        acts = activations[-1]
        grads = gradients[-1]

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1)
        cam = torch.relu(cam)
        cam = cam[0].detach().cpu().numpy()

        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        probability = float(
            torch.sigmoid(logit).detach().cpu()
        )

        return cam, probability

    finally:
        h1.remove()
        h2.remove()


# =========================================================
# HEADER
# =========================================================
st.markdown(
    """
    <div class="hero">
        <div class="hero-eyebrow">🏥 Clinical Decision-Support Prototype</div>
        <h1>Hospital Multi-Model AI Screening</h1>
        <p>Vital-Signs Risk Screening • Breast Cancer Detection • Explainable AI</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.markdown("## Hospital AI")
    st.caption("Decision-support screening prototype")

    if st.button("Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()

    st.divider()
    st.markdown("### System Modules")
    st.write("🫀 Vital-Signs Risk Screening")
    st.write("🩻 Breast Cancer Detection")
    st.write("🧠 Explainable AI")
    st.divider()
    st.markdown(
        "**Scope**\n\n"
        "This application is a screening/decision-support prototype. "
        "It is not a certified diagnostic device and does not replace "
        "clinical assessment."
    )

tab1, tab2 = st.tabs([
    "🫀 Vital-Signs Risk Screening",
    "🩻 Breast Cancer Detection",
])

# =========================================================
# TAB 1 — VITAL SIGNS
# =========================================================
with tab1:
    st.markdown('<div class="section-title">Vital-Signs Risk Screening</div>',
                unsafe_allow_html=True)

    st.markdown(
        '<div class="card"><b>PRD flow:</b> enter the patient vital signs → click <b>Predict</b> → '
        'the model returns the clinical risk classification with supporting feature importance.</div>',
        unsafe_allow_html=True,
    )

    model = load_vital_model()

    c1, c2 = st.columns([1.35, 1])

    with c1:
        st.markdown("### Patient / Measurement Inputs")
        st.caption("Enter the patient's vital signs below, then run the model.")

        # Compact real-data examples
        VITAL_EXAMPLES = {
            "Normal 1": {"hr": 94.0, "resp": 21.0, "spo2": 97.0, "temp": 36.2},
            "Normal 2": {"hr": 94.0, "resp": 25.0, "spo2": 97.0, "temp": 36.2},
            "Normal 3": {"hr": 93.0, "resp": 26.0, "spo2": 95.0, "temp": 37.0},
            "Abnormal 1": {"hr": 101.0, "resp": 25.0, "spo2": 93.0, "temp": 38.0},
            "Abnormal 2": {"hr": 55.0, "resp": 11.0, "spo2": 100.0, "temp": 35.0},
            "Abnormal 3": {"hr": 94.0, "resp": 26.0, "spo2": 97.0, "temp": 42.0},
        }

        # Initialize input values ONCE.
        # The examples update these session-state keys before the widgets
        # are created, so Streamlit will actually change the displayed numbers.
        if "vital_hr" not in st.session_state:
            st.session_state.vital_hr = 75.0
        if "vital_resp" not in st.session_state:
            st.session_state.vital_resp = 18.0
        if "vital_spo2" not in st.session_state:
            st.session_state.vital_spo2 = 98.0
        if "vital_temp" not in st.session_state:
            st.session_state.vital_temp = 36.8
        if "vital_example" not in st.session_state:
            st.session_state.vital_example = None

        st.caption("Quick Examples")
        ex_cols = st.columns(6)

        for col, name in zip(ex_cols, VITAL_EXAMPLES):
            with col:
                if st.button(
                    name,
                    key=f"vital_example_{name}",
                    use_container_width=True
                ):
                    example = VITAL_EXAMPLES[name]

                    # Update the actual number_input widget states.
                    st.session_state.vital_hr = example["hr"]
                    st.session_state.vital_resp = example["resp"]
                    st.session_state.vital_spo2 = example["spo2"]
                    st.session_state.vital_temp = example["temp"]
                    st.session_state.vital_example = name

                    # Rerun so the new values appear immediately.
                    st.rerun()

        a, b = st.columns(2)

        with a:
            hr = st.number_input(
                "Heart Rate (BPM)",
                min_value=0.0,
                max_value=300.0,
                step=1.0,
                key="vital_hr"
            )

            resp = st.number_input(
                "Respiratory Rate (BPM)",
                min_value=0.0,
                max_value=100.0,
                step=1.0,
                key="vital_resp"
            )

        with b:
            spo2 = st.number_input(
                "SpO₂ (%)",
                min_value=0.0,
                max_value=100.0,
                step=1.0,
                key="vital_spo2"
            )

            temp = st.number_input(
                "Temperature (°C)",
                min_value=25.0,
                max_value=45.0,
                step=0.1,
                key="vital_temp"
            )

        if st.session_state.vital_example:
            st.caption(
                f"Loaded example: **{st.session_state.vital_example}**"
            )

        analyze_vital = st.button(
            "Analyze Vital Signs",
            type="primary",
            use_container_width=True,
        )

    with c2:
        st.markdown("### Model Output")
        if analyze_vital:
            if model is None:
                st.warning(
                    "Vital-sign model file not found. Put the trained model at "
                    "`models/vital_signs_model.joblib`."
                )
            else:
                row = pd.DataFrame([{
                    "Heart_Rate": hr,
                    "Respiratory_Rate": resp,
                    "SpO2": spo2,
                    "Temperature": temp,
                }])

                try:
                    raw_pred = np.asarray(model.predict(row)).reshape(-1)[0]

                    # The training pipeline uses string labels: Normal / Abnormal.
                    # Also support models trained with encoded 0 / 1 labels.
                    if isinstance(raw_pred, (int, np.integer, float, np.floating)):
                        label = VITAL_LABELS.get(int(raw_pred), str(raw_pred))
                    else:
                        label = str(raw_pred)
                        if label.lower() == "0":
                            label = "Low Risk"
                        elif label.lower() == "1":
                            label = "High Risk"

                    if hasattr(model, "predict_proba"):
                        proba = np.asarray(model.predict_proba(row))[0]
                        classes = getattr(model, "classes_", None)

                        if classes is None and hasattr(model, "named_steps"):
                            final_estimator = list(model.named_steps.values())[-1]
                            classes = getattr(final_estimator, "classes_", None)

                        prob = None
                        if classes is not None:
                            classes = list(classes)
                            abnormal_idx = next(
                                (i for i, c in enumerate(classes)
                                 if str(c).lower() == "abnormal" or str(c) == "1"),
                                None,
                            )
                            if abnormal_idx is not None and abnormal_idx < len(proba):
                                prob = float(proba[abnormal_idx])
                        elif len(proba) == 2:
                            prob = float(proba[1])
                    else:
                        prob = None

                    cls = "status-normal" if label.lower() == "normal" else "status-abnormal"
                    st.markdown(
                        f'<div class="status {cls}">Predicted Risk Level: {label}</div>',
                        unsafe_allow_html=True,
                    )

                    if prob is not None:
                        if cls == "status-normal":
                            card_metric("Low Risk Probability", f"{1 - prob:.1%}")
                        else:
                            card_metric("High Risk Probability", f"{prob:.1%}")
                        st.caption("Risk labels are presented as Low Risk / High Risk for the PRD interface. The underlying model was trained on the dataset classes and this is a presentation mapping. Age, Gender and Blood Pressure are not used because they are not present in the training data.")

                    explanation = vital_explanation(
                        model,
                        [hr, resp, spo2, temp],
                    )
                    if explanation is not None:
                        chart_data = explanation[explanation["Feature"] != "Temperature"]
                        st.markdown("### Feature Importance")
                        st.bar_chart(
                            chart_data.set_index("Feature")["Importance"]
                        )
                        st.caption(
                            "Feature importance indicates model influence; it is not "
                            "causal medical evidence."
                        )

                except Exception as e:
                    st.error(f"Inference error: {e}")
        else:
            st.info("Enter the measurements and click **Analyze Vital Signs**.")

    st.markdown(
        '<div class="disclaimer"><b>Important:</b> The output is a model prediction '
        'for screening/decision support. It is not a diagnosis or treatment recommendation.</div>',
        unsafe_allow_html=True,
    )

# =========================================================
# TAB 2 — BREAST CANCER
# =========================================================
with tab2:
    st.markdown('<div class="section-title">Breast Cancer Detection</div>',
                unsafe_allow_html=True)

    st.markdown(
        '<div class="card"><b>Purpose:</b> analyze a mammogram ROI/POI image with '
        '<b>EfficientNet-B0</b> and return a Cancer / No Cancer prediction with '
        'confidence and Grad-CAM visualization.</div>',
        unsafe_allow_html=True,
    )

    breast_model, breast_meta = load_breast_model()

    left, right = st.columns([1.1, 1.2])

    with left:
        uploaded = st.file_uploader(
            "Upload Mammogram ROI / POI",
            type=["png", "jpg", "jpeg"],
            help="Use the ROI/POI image format represented by the trained model.",
        )

        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            st.image(
                image,
                caption="Uploaded ROI / POI",
                use_container_width=True,
            )

        analyze_breast = st.button(
            "Analyze Mammogram",
            type="primary",
            use_container_width=True,
        )

    with right:
        st.markdown("### Model Output")

        if analyze_breast:
            if uploaded is None:
                st.warning("Please upload a mammogram ROI/POI image first.")
            elif breast_model is None:
                st.warning(
                    "Breast-cancer model file not found or could not be loaded. "
                    "Put the trained multi-task checkpoint at "
                    "`models/breast_cancer_model.pt`."
                )
            else:
                try:
                    img_size = 512
                    threshold = 0.05

                    tfm = breast_transform(img_size)
                    tensor = tfm(image).unsqueeze(0)

                    # -------------------------------------------------
                    # One image -> Cancer + automatic View + Density
                    # -------------------------------------------------
                    breast_model.eval()

                    with torch.no_grad():
                        cancer_logits, view_logits, density_logits = breast_model(
                            tensor
                        )

                        probability = torch.sigmoid(
                            cancer_logits.squeeze(1)
                        ).item()

                        view_idx = view_logits.argmax(
                            dim=1
                        ).item()

                        density_idx = density_logits.argmax(
                            dim=1
                        ).item()

                    predicted_view = breast_meta["view_classes"][view_idx]
                    predicted_density = breast_meta["density_classes"][density_idx]

                    cancer = probability >= threshold
                    label = "Cancer" if cancer else "No Cancer"
                    status_class = (
                        "status-cancer"
                        if cancer
                        else "status-no-cancer"
                    )

                    st.markdown(
                        f'<div class="status {status_class}">Result: {label}</div>',
                        unsafe_allow_html=True,
                    )

                    m1, m2, m3 = st.columns(3)
                    with m1:
                        card_metric(
                            "Cancer probability",
                            f"{probability:.1%}",
                        )
                    with m2:
                        card_metric(
                            "View",
                            str(predicted_view),
                        )
                    with m3:
                        card_metric(
                            "Density",
                            str(predicted_density),
                        )

                    # -------------------------------------------------
                    # Grad-CAM — cancer head only
                    # -------------------------------------------------
                    tensor.requires_grad_(True)

                    target_layer = breast_model.backbone.features[-1]

                    cam, _ = gradcam(
                        breast_model,
                        tensor,
                        target_layer,
                    )

                    cam_img = np.array(
                        image.resize((img_size, img_size))
                    ).astype(np.float32) / 255.0

                    cam_uint8 = (
                        np.clip(cam, 0, 1) * 255
                    ).astype(np.uint8)

                    cam_resized = np.array(
                        Image.fromarray(
                            cam_uint8,
                            mode="L",
                        ).resize(
                            (img_size, img_size),
                            Image.Resampling.BILINEAR,
                        )
                    ).astype(np.float32) / 255.0

                    import matplotlib

                    cmap = matplotlib.colormaps["jet"]
                    heat = cmap(cam_resized)[..., :3]

                    overlay = (
                        0.55 * cam_img
                        + 0.45 * heat
                    )
                    overlay = np.clip(
                        overlay,
                        0,
                        1,
                    )

                    st.markdown(
                        "### Explainability — Grad-CAM"
                    )
                    st.image(
                        overlay,
                        caption="Model attention visualization",
                        use_container_width=True,
                    )
                    st.caption(
                        "View and Density are inferred automatically from "
                        "the uploaded image. Grad-CAM shows image regions "
                        "associated with the cancer model output; it does "
                        "not prove that a region is malignant."
                    )

                except Exception as e:
                    st.error(
                        f"Breast-cancer inference error: {e}"
                    )
        else:
            st.info(
                "Upload an image and click **Analyze Mammogram**."
            )

    st.markdown(
        '<div class="disclaimer"><b>Important:</b> This system is a research/prototype '
        'decision-support tool. It is not a certified medical diagnostic device and '
        'must not replace radiology, biopsy, or clinical assessment.</div>',
        unsafe_allow_html=True,
    )

# Footer
st.divider()
st.caption(
    "Hospital Multi-Model AI Risk Screening System • "
    "Vital Signs + Breast Cancer • Explainable AI • Prototype"
)
