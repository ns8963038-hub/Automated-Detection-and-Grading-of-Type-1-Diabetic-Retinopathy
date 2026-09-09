"""Streamlit demo: upload a fundus photograph, get a DR grade and an explanation.

Stage 7 of the system architecture, made interactive.

Run:  .venv/bin/streamlit run app/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# Make `src` importable when Streamlit runs this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config as C
from src.predict import GRADE_STATUS, available_models, load_model, predict

st.set_page_config(
    page_title="DR Grading",
    page_icon="👁",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------
# Design tokens
#
# The five DR grades are ordered tiers, so the confidence chart uses a single
# hue stepped light->dark (an ordinal ramp) rather than five unrelated colours.
# Both ramps were checked with a palette validator: monotone lightness,
# adjacent dL >= 0.06, single hue, surface-facing end clearing 2:1.
#
# Status colours appear only in the verdict, one at a time, and always beside
# the grade name -- colour never carries the meaning alone.
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Design tokens
#
# Structural colours (surfaces, borders, text) are deliberately theme-agnostic:
# translucent neutrals over whatever Streamlit paints, and `currentColor` for
# text. They therefore render correctly in light and dark without any theme
# detection, and a wrong guess can never produce invisible text.
#
# Only the ordinal ramp needs to know the theme. The five DR grades are ordered
# tiers, so the bars use one hue stepped light-to-dark. No single five-step
# ramp clears the 2:1 floor against both a white and a near-black surface -
# the usable band is too narrow - so each mode gets its own, and both were
# checked with a palette validator for monotone lightness, adjacent dL >= 0.06,
# single hue, and surface contrast.
# --------------------------------------------------------------------------
ORD_LIGHT = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
ORD_DARK = ["#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4"]


def theme_is_dark() -> bool:
    """Whether Streamlit is currently rendering dark.

    Two sources, and they disagree in a way that matters:

    * `st.get_option("theme.base")` is the *configured* theme. It is None when
      the app has not set one, and wins whenever it is set.
    * `st.context.theme` reports the *browser's* preference, which Streamlit
      follows only when nothing is configured. Reading this one alone paints
      dark components onto a light page whenever the user's OS is dark but the
      app is configured light.
    """
    try:
        base = st.get_option("theme.base")
        if isinstance(base, str) and base.lower() in ("light", "dark"):
            return base.lower() == "dark"
    except Exception:
        pass
    try:
        theme = st.context.theme
        kind = (theme.get("type") if isinstance(theme, dict)
                else getattr(theme, "type", None))
        return kind == "dark"
    except Exception:
        return False


def build_css(dark: bool) -> str:
    ramp = ORD_DARK if dark else ORD_LIGHT
    ords = "".join(f"  --ord-{i + 1}: {c};\n" for i, c in enumerate(ramp))
    return f"""
<style>
.dr {{
  /* Neutrals that read correctly on any background Streamlit provides. */
  --raised:  color-mix(in srgb, currentColor 5%, transparent);
  --border:  color-mix(in srgb, currentColor 22%, transparent);
  --track:   color-mix(in srgb, currentColor 16%, transparent);
  --ink:     currentColor;
  --ink-2:   color-mix(in srgb, currentColor 72%, transparent);
  --ink-3:   color-mix(in srgb, currentColor 52%, transparent);
{ords}}}

/* ---------- verdict ---------- */
.dr-verdict {{
  display: flex; align-items: center; gap: 20px;
  padding: 22px 26px; border-radius: 12px;
  background: color-mix(in srgb, var(--accent) 10%, transparent);
  border: 1px solid color-mix(in srgb, var(--accent) 38%, transparent);
  border-left: 7px solid var(--accent);
}}
.dr-dot {{
  width: 54px; height: 54px; border-radius: 50%;
  background: var(--accent); flex-shrink: 0;
  display: flex; align-items: center; justify-content: center;
  color: #fff; font-size: 24px; font-weight: 800;
  font-variant-numeric: tabular-nums;
}}
.dr-verdict-body {{ flex: 1; min-width: 0; }}
.dr-grade {{
  font-size: 25px; font-weight: 700; color: var(--ink);
  line-height: 1.15; letter-spacing: -0.01em;
}}
.dr-action {{ font-size: 14px; color: var(--ink-2); margin-top: 5px; }}

/* ---------- referable banner ---------- */
.dr-refer {{
  display: flex; align-items: center; gap: 10px;
  margin-top: 12px; padding: 11px 16px; border-radius: 9px;
  font-size: 14px; font-weight: 600; color: var(--ink);
  background: color-mix(in srgb, var(--accent) 7%, transparent);
  border: 1px solid color-mix(in srgb, var(--accent) 26%, transparent);
}}
.dr-refer .tag {{
  font-size: 11px; font-weight: 800; letter-spacing: 0.07em;
  text-transform: uppercase; padding: 3px 9px; border-radius: 5px;
  background: var(--accent); color: #fff; flex-shrink: 0;
}}

/* ---------- confidence bars ---------- */
.dr-row {{ display: flex; align-items: center; gap: 12px; margin: 9px 0; }}
.dr-name {{
  width: 148px; flex-shrink: 0; text-align: right;
  font-size: 13px; color: var(--ink-2);
}}
.dr-track {{
  flex: 1; min-width: 50px; height: 17px;
  background: var(--track); border-radius: 4px; overflow: hidden;
}}
.dr-fill {{ height: 100%; border-radius: 4px; }}
.dr-val {{
  width: 56px; flex-shrink: 0; font-size: 13px;
  color: var(--ink-2); font-variant-numeric: tabular-nums;
}}
.dr-row.pred .dr-name, .dr-row.pred .dr-val {{
  font-weight: 700; color: var(--ink);
}}

/* ---------- legend ---------- */
.dr-legend {{ display: flex; align-items: center; gap: 10px; margin: 8px 0; }}
.dr-chip {{
  width: 12px; height: 12px; border-radius: 50%;
  background: var(--accent); flex-shrink: 0;
}}
.dr-legend span {{ font-size: 13px; color: var(--ink-2); }}

/* ---------- findings ---------- */
.dr-find {{
  display: flex; align-items: baseline; gap: 12px; padding: 10px 0;
  border-bottom: 1px solid var(--border);
}}
.dr-find:last-child {{ border-bottom: 0; }}
.dr-find .sw {{
  width: 11px; height: 11px; border-radius: 3px;
  background: var(--accent); flex-shrink: 0; align-self: center;
}}
.dr-find .nm {{ font-size: 14px; font-weight: 600; color: var(--ink); flex: 1; }}
.dr-find .ct {{
  font-size: 13px; color: var(--ink-2); font-variant-numeric: tabular-nums;
  width: 78px; text-align: right;
}}
.dr-find .en {{
  font-size: 13px; font-weight: 700; width: 96px; text-align: right;
  font-variant-numeric: tabular-nums; color: var(--ink);
}}
.dr-why {{ font-size: 12.5px; color: var(--ink-3); margin: 2px 0 0 23px; }}

/* ---------- misc ---------- */
.dr-step {{
  font-size: 12px; color: var(--ink-3); text-transform: uppercase;
  letter-spacing: 0.08em; font-weight: 700; margin-bottom: 3px;
}}
.dr-empty {{
  padding: 34px; border-radius: 12px; text-align: center;
  border: 2px dashed var(--border); background: var(--raised);
}}
.dr-empty h3 {{ color: var(--ink); font-size: 17px; margin: 0 0 7px; }}
.dr-empty p  {{ color: var(--ink-2); font-size: 14px; margin: 0; }}
.dr-note {{
  font-size: 12.5px; color: var(--ink-3); line-height: 1.55;
  border-left: 3px solid var(--border); padding-left: 12px;
}}
</style>
"""


def verdict(res: dict) -> str:
    refer_tag = "Referable" if res["referable"] else "Monitor"
    refer_text = (
        "Grade 2 or above - refer to an ophthalmologist."
        if res["referable"]
        else "Below the referral threshold - routine screening."
    )
    return f"""
<div class="dr" style="--accent:{res['color']}">
  <div class="dr-verdict">
    <div class="dr-dot">{res['grade']}</div>
    <div class="dr-verdict-body">
      <div class="dr-grade">{res['short_label']}</div>
      <div class="dr-action">{res['action']}</div>
    </div>
  </div>
  <div class="dr-refer">
    <span class="tag">{refer_tag}</span><span>{refer_text}</span>
  </div>
</div>
"""


def confidence_bars(probs: list[float], predicted: int) -> str:
    rows = []
    for grade, p in enumerate(probs):
        cls = " pred" if grade == predicted else ""
        rows.append(
            f'<div class="dr-row{cls}" title="{C.CLASS_NAMES[grade]}: {p:.1%}">'
            f'<div class="dr-name">{C.CLASS_NAMES[grade]}</div>'
            f'<div class="dr-track"><div class="dr-fill" '
            f'style="width:{max(p * 100, 0.7):.1f}%;background:var(--ord-{grade + 1})">'
            f"</div></div>"
            f'<div class="dr-val">{p:.1%}</div></div>'
        )
    return f'<div class="dr">{"".join(rows)}</div>'


def legend_row(grade: int) -> str:
    return (
        f'<div class="dr" style="--accent:{GRADE_STATUS[grade]["color"]}">'
        f'<div class="dr-legend"><div class="dr-chip"></div>'
        f"<span>{C.CLASS_NAMES[grade]}</span></div></div>"
    )


def caption(step: str, text: str) -> str:
    return f'<div class="dr"><div class="dr-step">{step}</div>' \
           f'<div style="font-size:13px;color:var(--ink-2)">{text}</div></div>'


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown(build_css(theme_is_dark()), unsafe_allow_html=True)
st.title("Diabetic Retinopathy Grading")
st.caption(
    "Convolutional neural network trained on the APTOS 2019 fundus dataset. "
    "Research prototype for academic assessment - not a medical device."
)

trained = available_models()
if not trained:
    st.error("No trained model found.")
    st.code("python -m src.train --model efficientnet", language="bash")
    st.stop()

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.subheader("Model")
    model_key = st.selectbox(
        "Checkpoint", trained,
        index=trained.index(C.DEFAULT_MODEL) if C.DEFAULT_MODEL in trained else 0,
        label_visibility="collapsed",
    )
    _, ckpt, device = load_model(model_key)
    vm = ckpt["val_metrics"]

    c1, c2 = st.columns(2)
    c1.metric("Val QWK", f"{vm['qwk']:.3f}")
    c2.metric("Val acc", f"{vm['accuracy']:.1%}")
    st.caption(
        f"`{ckpt['arch']}` · epoch {ckpt['epoch']} · `{device}`"
    )

    st.divider()
    st.subheader("Options")
    show_cam = st.toggle(
        "Grad-CAM explanation", value=True,
        help="Highlight the retinal regions that drove the predicted grade.",
    )
    show_lesions = st.toggle(
        "Lesion analysis", value=True,
        help="Detect the retinal features a clinician grades on, and measure "
             "how concentrated each one is where the model looked.",
    )

    st.divider()
    st.subheader("Severity scale")
    for g in range(C.NUM_CLASSES):
        st.markdown(legend_row(g), unsafe_allow_html=True)
    st.markdown(
        '<div class="dr"><div class="dr-note" style="margin-top:10px">'
        "Grade 2 and above is <b>referable DR</b> - the threshold at which a "
        "screening programme refers a patient to an ophthalmologist."
        "</div></div>",
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------
# Input
# --------------------------------------------------------------------------
uploaded = st.file_uploader(
    "Retinal fundus image", type=["png", "jpg", "jpeg"],
    help="The same preprocessing used during training is applied automatically.",
)

use_sample, sample_path = False, None
if not uploaded and C.IMAGES_RAW.exists():
    samples = sorted(C.IMAGES_RAW.glob("*.png"))[:12]
    if samples:
        pick = st.selectbox(
            "...or try a sample from the dataset",
            ["-"] + [p.name for p in samples],
        )
        if pick != "-":
            sample_path, use_sample = C.IMAGES_RAW / pick, True

if not uploaded and not use_sample:
    st.markdown(
        '<div class="dr"><div class="dr-empty">'
        "<h3>Upload a fundus photograph to begin</h3>"
        "<p>The image is cropped, contrast-enhanced and masked, then graded "
        "0-4 on the international DR severity scale.</p>"
        "</div></div>",
        unsafe_allow_html=True,
    )
    st.stop()

# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------
if use_sample:
    raw = cv2.cvtColor(cv2.imread(str(sample_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
else:
    decoded = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        st.error("Could not decode that file as an image.")
        st.stop()
    raw = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)

with st.spinner("Analysing retinal image..."):
    res = predict(raw, model_key=model_key, with_cam=show_cam,
                  with_lesions=show_lesions)

# --- result first: it is what the user came for ------------------------
left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("Result")
    st.markdown(verdict(res), unsafe_allow_html=True)
    # "Grade 3 of 4" reads like a score, so show the runner-up instead: on an
    # ordinal task the second-choice grade says more about the model's
    # certainty than restating the prediction does.
    probs = res["probabilities"]
    order = sorted(range(len(probs)), key=lambda g: probs[g], reverse=True)
    runner_up = order[1]
    margin = probs[order[0]] - probs[runner_up]

    m1, m2 = st.columns(2)
    m1.metric("Confidence", f"{res['confidence']:.1%}",
              help="Softmax probability assigned to the predicted grade.")
    # No `delta` here: Streamlit renders it with a up/down arrow, which would
    # imply the runner-up is rising rather than trailing.
    # The value is the grade number, not the name: st.metric renders values in
    # a large font and "Proliferative" truncates at this column width.
    m2.metric("Next most likely", f"Grade {runner_up}",
              help="A narrow margin means the model nearly chose this grade "
                   "instead.")
    m2.caption(f"{C.CLASS_SHORT[runner_up]} · {probs[runner_up]:.1%} "
               f"({margin:.0%} behind)")

with right:
    st.subheader("Confidence across grades")
    st.markdown(confidence_bars(res["probabilities"], res["grade"]),
                unsafe_allow_html=True)
    with st.expander("View as table"):
        st.dataframe(
            pd.DataFrame({
                "Grade": C.CLASS_NAMES,
                "Probability": [f"{p:.2%}" for p in res["probabilities"]],
            }),
            hide_index=True, width="stretch",
        )

st.divider()

# --- how the image was processed ---------------------------------------
st.subheader("How the image was processed")
cols = st.columns(3 if show_cam else 2, gap="medium")

cols[0].image(raw, width="stretch")
cols[0].markdown(caption("Step 1", "Original upload, as provided."),
                 unsafe_allow_html=True)

cols[1].image(res["processed"], width="stretch")
cols[1].markdown(
    caption("Step 2", "Border cropped, contrast enhanced (Ben Graham), "
                      "circular field-of-view mask applied."),
    unsafe_allow_html=True,
)

if show_cam:
    cols[2].image(res["overlay"], width="stretch")
    cols[2].markdown(
        caption("Step 3", "Grad-CAM: warmer regions raised the score for the "
                          "predicted grade."),
        unsafe_allow_html=True,
    )

    with st.expander("What is Grad-CAM?"):
        st.markdown(
            "Grad-CAM traces the predicted grade's score back through the "
            "network to its last convolutional layer, so the heatmap shows "
            "**which parts of the retina pushed the prediction up**.\n\n"
            "On a higher grade you should expect heat over haemorrhages, "
            "hard exudates and neovascularisation. On a healthy retina the "
            "map is usually diffuse, because there is no lesion to point at. "
            "This turns a black-box score into something a clinician can "
            "argue with."
        )

# --- what the model was looking at ---------------------------------------
if show_lesions and res.get("lesions"):
    st.divider()
    st.subheader("What is in the region the model focused on")

    lesions = res["lesions"]
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.image(res["lesion_overlay"], width="stretch")
        st.markdown(
            caption("Detected features",
                    "Red = microaneurysms and haemorrhages. "
                    "Yellow = hard exudates. Blue = vasculature, shown for "
                    "orientation."),
            unsafe_allow_html=True,
        )

    with right:
        if lesions["sparse"]:
            st.markdown(
                f'<div class="dr"><div class="dr-note">'
                f"Only {lesions['total_lesions']} candidate lesions were found. "
                f"For a low grade that is the explanation. For a high one it "
                f"means the model is relying on features these detectors do "
                f"not capture - subtle texture, colour or vessel calibre - so "
                f"read the Grad-CAM map rather than this panel."
                f"</div></div>",
                unsafe_allow_html=True,
            )
        else:
            rows = []
            for f in lesions["findings"]:
                enrich = f.get("enrichment", 0.0)
                verdict = ("concentrated here" if enrich >= 1.35
                           else "evenly spread" if enrich >= 0.75
                           else "mostly elsewhere")
                rgb = f["colour"]
                rows.append(
                    f'<div class="dr-find" style="--accent:rgb{rgb}">'
                    f'<div class="sw"></div>'
                    f'<div class="nm">{f["short"]}</div>'
                    f'<div class="ct">{f["count"]} found</div>'
                    f'<div class="en">{enrich:.1f}x</div></div>'
                    f'<div class="dr-why">{verdict} &middot; {f["meaning"]}</div>'
                )
            st.markdown(f'<div class="dr">{"".join(rows)}</div>',
                        unsafe_allow_html=True)

        with st.expander("How to read this"):
            st.markdown(
                "**The multiplier** is how concentrated a feature is inside "
                "the quarter of the retina the model attended to, against its "
                "average density over the whole retina. `2.0x` means twice as "
                "dense there as elsewhere; `1.0x` means no more than you would "
                "expect by area.\n\n"
                "**The optic disc is excluded** from both detectors. It is "
                "naturally bright and yellow, so it would otherwise dominate "
                "the exudate count on every single image."
            )

st.divider()
st.markdown(
    f'<div class="dr"><div class="dr-note">'
    f"Model <b>{res['arch']}</b>, run <b>{res['model_key']}</b>. "
    f"This tool supports screening decisions; it does not replace an "
    f"ophthalmologist's diagnosis."
    f"</div></div>",
    unsafe_allow_html=True,
)
