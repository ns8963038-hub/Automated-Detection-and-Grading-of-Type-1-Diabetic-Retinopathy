"""Single-image inference: the runtime path behind the demo app.

Wraps stages 2 -> 7 for one uploaded fundus photograph:
preprocess, forward pass, softmax, Grad-CAM overlay.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from src import config as C
from src.dataset import eval_transform
from src.gradcam import GradCAM, overlay_heatmap
from src.lesions import explain as explain_lesions, overlay_lesions
from src.models import build_model, get_device
from src.preprocess import preprocess_image

# Clinical severity mapping. Status colors are used one at a time as a badge
# and are always paired with the icon and the grade name, never colour alone.
GRADE_STATUS = {
    0: {"role": "good",     "color": "#0ca30c", "icon": "OK",
        "action": "No diabetic retinopathy detected. Routine annual screening."},
    1: {"role": "warning",  "color": "#fab219", "icon": "!",
        "action": "Mild non-proliferative DR. Re-screen in 12 months."},
    2: {"role": "serious",  "color": "#ec835a", "icon": "!!",
        "action": "Moderate NPDR. Referable - ophthalmologist review advised."},
    3: {"role": "critical", "color": "#d03b3b", "icon": "!!!",
        "action": "Severe NPDR. Referable - prompt specialist referral."},
    4: {"role": "critical", "color": "#d03b3b", "icon": "!!!!",
        "action": "Proliferative DR. Urgent ophthalmology referral."},
}


@lru_cache(maxsize=4)
def load_model(model_key: str = C.DEFAULT_MODEL):
    """Load a trained checkpoint. Cached so Streamlit reruns stay fast."""
    device = get_device()
    ckpt_path = C.MODEL_DIR / f"{model_key}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No trained model at {ckpt_path}.\n"
            f"Train one first:  python -m src.train --model {model_key}"
        )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    # Architecture and activation come from the checkpoint, so ablation runs
    # with names like `baseline_sgd_gelu` load correctly.
    arch_key = ckpt.get("model_key", model_key)
    activation = ckpt.get("config", {}).get("activation", C.ACTIVATION)
    model = build_model(arch_key, pretrained=False, activation=activation)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt, device


def available_models() -> list[str]:
    """Every trained run on disk, default model keys first."""
    runs = sorted(p.stem[:-5] for p in C.MODEL_DIR.glob("*_best.pt"))
    return sorted(runs, key=lambda r: (r not in C.MODELS, r))


def predict(image_rgb: np.ndarray, model_key: str = C.DEFAULT_MODEL,
            with_cam: bool = True, with_lesions: bool = False) -> dict:
    """Grade one fundus image.

    Args:
        image_rgb: raw uploaded image as an HxWx3 RGB uint8 array.
        model_key: which trained checkpoint to use.
        with_cam:  also compute the Grad-CAM explanation.
        with_lesions: also run the classical lesion detectors and report how
            strongly each type is enriched inside the attended region.

    Returns a dict with the predicted grade, per-class probabilities, the
    preprocessed image, and (optionally) the heatmap overlay.
    """
    model, ckpt, device = load_model(model_key)
    img_size = ckpt["img_size"]

    # Stage 2: identical preprocessing to training.
    processed = preprocess_image(image_rgb, size=C.PREPROCESS_SIZE)
    tensor = eval_transform(img_size)(image=processed)["image"]
    tensor = tensor.unsqueeze(0).to(device)

    result = {
        "processed": processed,
        "model_key": model_key,
        "arch": ckpt["arch"],
    }

    if with_cam:
        with GradCAM(model) as cam:
            heatmap, pred, probs = cam(tensor)
        # Overlay onto the preprocessed view so heat lines up with what the
        # network actually saw.
        result["overlay"] = overlay_heatmap(processed, heatmap)
        result["heatmap"] = heatmap
    else:
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            pred = int(probs.argmax())

    if with_lesions:
        # Detectors run on the original image, not the preprocessed one: the
        # training-time enhancement recentres every image on mid-grey, which
        # destroys the hue difference between yellow exudates and red
        # haemorrhages that the detectors rely on.
        lesions = explain_lesions(image_rgb, result.get("heatmap"))
        result["lesions"] = lesions
        result["lesion_overlay"] = overlay_lesions(lesions["image"], lesions["masks"])

    result.update({
        "grade": int(pred),
        "label": C.CLASS_NAMES[pred],
        "short_label": C.CLASS_SHORT[pred],
        "confidence": float(probs[pred]),
        "probabilities": probs.astype(float).tolist(),
        "referable": bool(pred >= 2),
        **GRADE_STATUS[int(pred)],
    })
    return result


def predict_file(path: str | Path, model_key: str = C.DEFAULT_MODEL,
                 with_cam: bool = True, with_lesions: bool = False) -> dict:
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return predict(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), model_key,
                   with_cam, with_lesions)
