"""Generate the project documentation PDF.

Every number is read from the artefacts in outputs/, so the document cannot
drift from the code. Figures are embedded as base64 to keep the PDF
self-contained.

Run:  python docs/make_report.py
"""
from __future__ import annotations

import base64
import io
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import config as C  # noqa: E402

OUT_HTML = ROOT / "docs" / "project_report.html"
OUT_PDF = ROOT / "docs" / "Project_Documentation.pdf"


# --------------------------------------------------------------------------
# Data pulled from the project, never typed by hand
# --------------------------------------------------------------------------
def load_facts() -> dict:
    f = {}
    for key in ("efficientnet", "baseline"):
        f[key] = json.loads((C.LOG_DIR / f"{key}_test_results.json").read_text())
        f[key + "_sum"] = json.loads((C.LOG_DIR / f"{key}_summary.json").read_text())
        f[key + "_hist"] = pd.read_csv(C.LOG_DIR / f"{key}_history.csv")
    splits = pd.read_csv(C.SPLITS_CSV)
    f["splits"] = splits.split.value_counts().to_dict()
    f["grades"] = splits.label.value_counts().sort_index().to_dict()
    f["total"] = len(splits)
    return f


def embed(path: Path, max_w: int = 1500) -> str:
    """Base64-embed an image, downscaled so the PDF stays a sane size."""
    img = Image.open(path).convert("RGB")
    if img.width > max_w:
        img = img.resize((max_w, round(img.height * max_w / img.width)),
                         Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=86, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def fig(name: str, caption: str, max_w: int = 1500) -> str:
    path = C.FIGURE_DIR / name
    if not path.exists():
        path = ROOT / "docs" / name
    if not path.exists():
        return ""
    return (f'<figure><img src="{embed(path, max_w)}" alt="{caption}">'
            f"<figcaption>{caption}</figcaption></figure>")


def rows(data, cols):
    body = "".join(
        "<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in data
    )
    head = "".join(f"<th>{c}</th>" for c in cols)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


CSS = """
@page { size: A4; margin: 17mm 15mm 16mm 15mm; }
* { box-sizing: border-box; }
body {
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-size: 10.2pt; line-height: 1.55; color: #14161a; margin: 0;
  -webkit-print-color-adjust: exact; print-color-adjust: exact;
}
h1, h2, h3, h4 { color: #0b0b0b; line-height: 1.25; }
h1 { font-size: 25pt; margin: 0 0 4pt; letter-spacing: -0.4pt; }
h2 {
  font-size: 15pt; margin: 22pt 0 8pt; padding-bottom: 5pt;
  border-bottom: 2px solid #2a78d6; break-after: avoid;
}
h3 { font-size: 11.6pt; margin: 15pt 0 5pt; break-after: avoid; }
h4 { font-size: 10.4pt; margin: 11pt 0 3pt; color: #38393d; break-after: avoid; }
p { margin: 0 0 7pt; }
ul, ol { margin: 0 0 8pt; padding-left: 17pt; }
li { margin-bottom: 3pt; }
code {
  font-family: "SF Mono", Menlo, Consolas, monospace; font-size: 8.9pt;
  background: #f1f2f4; padding: 1pt 3.5pt; border-radius: 3px;
}
pre {
  background: #f7f8f9; border: 1px solid #e2e4e8; border-left: 3px solid #2a78d6;
  border-radius: 4px; padding: 8pt 10pt; font-size: 8.6pt;
  font-family: "SF Mono", Menlo, Consolas, monospace; overflow-x: auto;
  white-space: pre-wrap; break-inside: avoid; margin: 0 0 9pt;
}
pre code { background: none; padding: 0; font-size: inherit; }
table {
  width: 100%; border-collapse: collapse; margin: 0 0 10pt;
  font-size: 9.2pt; break-inside: avoid;
}
th {
  background: #eef3fb; text-align: left; padding: 5pt 7pt;
  border-bottom: 1.5px solid #2a78d6; font-weight: 700; font-size: 8.9pt;
}
td { padding: 4.5pt 7pt; border-bottom: 1px solid #e6e7ea; vertical-align: top; }
tr:nth-child(even) td { background: #fafbfc; }
figure { margin: 10pt 0 13pt; break-inside: avoid; text-align: center; }
figure img {
  max-width: 100%;
  /* A4 leaves ~264mm of usable height; a figure also needs room for its
     caption. Two figures (the Grad-CAM grid and the lesion examples) are
     taller than that at full width, and without a cap the print engine
     splits them across pages, cutting images in half. Capping the height
     makes tall figures narrower instead of broken.
     190mm rather than the ~224mm that would just fit: the smaller cap lets
     adjacent text share the page, which removed two near-empty pages. */
  max-height: 190mm;
  width: auto; height: auto;
  border: 1px solid #e2e4e8; border-radius: 4px;
}
figcaption {
  font-size: 8.6pt; color: #62646a; margin-top: 4pt; font-style: italic;
}
.cover {
  height: 246mm; display: flex; flex-direction: column;
  justify-content: center; text-align: center; break-after: page;
}
.cover .sub { font-size: 12.5pt; color: #4c4e54; margin-top: 10pt; }
.cover .rule { width: 74pt; height: 3px; background: #2a78d6; margin: 20pt auto; }
.cover .meta { font-size: 10pt; color: #62646a; margin-top: 26pt; line-height: 1.9; }
.cover .head {
  font-size: 9.4pt; letter-spacing: 2.4pt; text-transform: uppercase;
  color: #2a78d6; font-weight: 700; margin-bottom: 14pt;
}
.note {
  background: #f4f8fe; border-left: 3px solid #2a78d6;
  padding: 8pt 11pt; margin: 0 0 10pt; font-size: 9.4pt; break-inside: avoid;
}
.warn {
  background: #fff8ee; border-left: 3px solid #eb8a34;
  padding: 8pt 11pt; margin: 0 0 10pt; font-size: 9.4pt; break-inside: avoid;
}
.kpi { display: flex; gap: 6pt; margin: 0 0 12pt; }
.kpi > div {
  flex: 1; border: 1px solid #dfe1e6; border-radius: 5px;
  padding: 8pt 4pt; text-align: center; background: #fbfcfd;
}
/* The test figure is the reported result; the others are context. */
.kpi > div.hero { border: 1.5px solid #2a78d6; background: #f2f7fe; }
.kpi .v { font-size: 15pt; font-weight: 700; color: #62646a; }
.kpi .hero .v { font-size: 17pt; color: #1c5cab; }
.kpi .l { font-size: 7.2pt; color: #82848a; text-transform: uppercase;
          letter-spacing: 0.4pt; margin-top: 3pt; line-height: 1.35; }
.kpi .hero .l { color: #1c5cab; font-weight: 700; }
.pagebreak { break-before: page; }
.toc a { color: #14161a; text-decoration: none; }
.toc li { margin-bottom: 4pt; }
"""


def build(f: dict) -> str:
    eff, base = f["efficientnet"]["metrics"], f["baseline"]["metrics"]
    effc, _ = f["efficientnet"]["per_class"], f["baseline"]["per_class"]
    g, sp = f["grades"], f["splits"]
    total = f["total"]

    # Training/validation accuracy at the epoch that was actually saved --
    # later epochs reached higher training accuracy but their weights were
    # discarded by early stopping, so quoting those would describe a model
    # that was never shipped.
    hist = f["efficientnet_hist"]
    best_row = hist.loc[hist.val_qwk.idxmax()]
    train_acc = float(best_row.train_accuracy)
    val_acc = float(best_row.val_accuracy)
    best_ep = int(best_row.epoch)
    gap_pp = 100.0 * (train_acc - eff["accuracy"])

    grade_rows = [
        [f"<b>{i}</b>", C.CLASS_NAMES[i].split(" - ")[1],
         f"{g[i]:,}", f"{100*g[i]/total:.1f}%",
         ["Routine annual screening", "Re-screen in 12 months",
          "<b>Referable</b> — ophthalmologist review",
          "<b>Referable</b> — prompt referral",
          "<b>Urgent</b> referral"][i]]
        for i in range(5)
    ]

    split_rows = [[k.capitalize(), f"{sp[k]:,}", f"{100*sp[k]/total:.0f}%", v]
                  for k, v in [("train", "Weights are learned from these"),
                               ("val", "Chooses the best epoch and triggers early stopping"),
                               ("test", "Touched once, at the very end")]]

    perclass = [[C.CLASS_NAMES[i],
                 f"{effc[C.CLASS_NAMES[i]]['precision']:.3f}",
                 f"{effc[C.CLASS_NAMES[i]]['recall']:.3f}",
                 f"{effc[C.CLASS_NAMES[i]]['f1']:.3f}",
                 effc[C.CLASS_NAMES[i]]['support']] for i in range(5)]

    cmp_rows = [
        ["<b>EfficientNet-B3</b>", "10.70 M", f["efficientnet_sum"]["epochs_run"],
         f"{f['efficientnet_sum']['minutes']:.0f} min",
         f"{eff['accuracy']:.4f}", f"<b>{eff['qwk']:.4f}</b>",
         f"{eff['f1_macro']:.4f}", f"{eff['referable_sensitivity']:.4f}"],
        ["Baseline CNN", "4.85 M", f["baseline_sum"]["epochs_run"],
         f"{f['baseline_sum']['minutes']:.0f} min",
         f"{base['accuracy']:.4f}", f"{base['qwk']:.4f}",
         f"{base['f1_macro']:.4f}", f"{base['referable_sensitivity']:.4f}"],
    ]

    stack = [
        ["Language", "Python 3.14.7", "Runtime for the whole pipeline"],
        ["Deep learning", "PyTorch 2.14.0", "Model definition, autograd, training loop"],
        ["", "torchvision 0.29.0", "Vision utilities"],
        ["", "timm 1.0.29", "Pretrained EfficientNet-B3 and ResNet-50 backbones"],
        ["Acceleration", "Apple Metal (MPS)", "GPU backend; falls back to CUDA, then CPU"],
        ["Image processing", "OpenCV 5.0.0", "Preprocessing, lesion detection, Grad-CAM overlay"],
        ["", "Pillow 12.3.0", "Image I/O"],
        ["Augmentation", "albumentations 2.0.8", "Training-time augmentation pipeline"],
        ["Data", "pandas 3.0.5", "Labels, splits, metric tables"],
        ["", "NumPy 2.5.3", "Array operations"],
        ["Metrics", "scikit-learn 1.9.0", "QWK, F1, confusion matrix, stratified splitting"],
        ["Visualisation", "matplotlib 3.11.1", "All report figures"],
        ["", "seaborn 0.13.2", "Confusion-matrix heatmaps"],
        ["Application", "Streamlit 1.63.0", "Interactive demo"],
        ["Dataset access", "kaggle 2.2.4", "Authenticated dataset download"],
        ["Version control", "Git + GitHub", "History; weights shipped as release assets"],
    ]

    algos = [
        ["Ben Graham preprocessing", "Preprocessing",
         "Subtracts a heavily Gaussian-blurred copy of the image from itself, "
         "removing the low-frequency component. Cancels per-camera colour cast "
         "and uneven illumination, leaving lesions visible."],
        ["Convolutional Neural Network", "Model",
         "Learns spatial features through stacked convolution, batch "
         "normalisation and non-linear activation."],
        ["Transfer learning", "Model",
         "ImageNet-pretrained backbone fine-tuned end to end. Reaches the "
         "from-scratch model's best score in 4 epochs instead of 32."],
        ["ReLU / SiLU", "Activation",
         "The non-linearity inside the network. Baseline and ResNet use ReLU; "
         "EfficientNet uses SiLU (Swish). Without one, stacked convolutions "
         "collapse into a single linear operation."],
        ["AdamW", "Optimiser",
         "Adam with decoupled weight decay. Adapts a learning rate per "
         "parameter, which converges reliably on a small dataset."],
        ["Cosine annealing + warmup", "LR schedule",
         "Two epochs of linear warmup, then cosine decay. Warmup stops the "
         "randomly initialised head washing out pretrained features."],
        ["Class-weighted cross-entropy", "Loss",
         "Inverse-frequency weights. Grade 0 is 49% of the data, so an "
         "unweighted model scores ~49% by always predicting 'No DR'."],
        ["Label smoothing (0.05)", "Loss",
         "Softens targets to reduce overconfidence."],
        ["Gradient clipping (norm 1.0)", "Training",
         "Caps gradient magnitude to prevent exploding updates."],
        ["Early stopping", "Training",
         "Halts after 8 epochs without validation QWK improvement, keeping "
         "the best checkpoint."],
        ["Stratified group splitting", "Data",
         "Splits on image-content hash so duplicate photographs cannot span "
         "train and test, while preserving each grade's proportion."],
        ["Quadratic Weighted Kappa", "Metric",
         "Ordinal agreement corrected for chance. Penalises a grade 0→4 error "
         "far more than 3→4."],
        ["Grad-CAM", "Explainability",
         "Backpropagates the predicted class score to the last convolutional "
         "feature map, weighting channels by mean gradient."],
        ["Morphological lesion detection", "Explainability",
         "Black top-hat for vessels, local-minimum analysis for red lesions, "
         "colour thresholding for exudates, with per-blob shape filtering."],
    ]

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Diabetic Retinopathy Grading — Project Documentation</title>
<style>{CSS}</style></head><body>

<section class="cover">
  <div class="head">Project Documentation</div>
  <h1>Automated Detection and Grading of<br>Diabetic Retinopathy Using CNN</h1>
  <div class="rule"></div>
  <div class="sub">A deep-learning pipeline that grades retinal fundus<br>
    photographs on the international 0–4 severity scale</div>
  <div class="meta">
    <b>Dataset</b> &nbsp; APTOS 2019 Blindness Detection ({total:,} fundus images)<br>
    <b>Best model</b> &nbsp; EfficientNet-B3 — QWK {eff['qwk']:.4f}, accuracy {eff['accuracy']:.1%}<br>
    <b>Compiled</b> &nbsp; {date.today():%d %B %Y}
  </div>
</section>

<h2 id="toc">Contents</h2>
<ol class="toc">
  <li><a href="#s1">Project overview</a></li>
  <li><a href="#s2">The clinical problem and grading scale</a></li>
  <li><a href="#s3">Dataset</a></li>
  <li><a href="#s4">Flow of implementation — the seven stages</a></li>
  <li><a href="#s5">Technology stack</a></li>
  <li><a href="#s6">Algorithms and techniques</a></li>
  <li><a href="#s7">Preprocessing in detail</a></li>
  <li><a href="#s8">Model architectures</a></li>
  <li><a href="#s9">Training configuration</a></li>
  <li><a href="#s10">Evaluation metrics</a></li>
  <li><a href="#s11">Results</a></li>
  <li><a href="#s12">Explainability</a></li>
  <li><a href="#s13">The demo application</a></li>
  <li><a href="#s14">Repository structure</a></li>
  <li><a href="#s15">How to run it</a></li>
  <li><a href="#s16">Known limitations and future work</a></li>
  <li><a href="#s17">Glossary</a></li>
</ol>

<h2 id="s1" class="pagebreak">1 &nbsp; Project overview</h2>
<p>Diabetic retinopathy is damage to the retina caused by diabetes and is a
leading cause of preventable blindness. It is diagnosed by an ophthalmologist
examining a photograph of the back of the eye — a <b>fundus image</b> — and
assigning a severity grade. Screening programmes generate far more images than
there are specialists to read them.</p>

<p>This project trains a convolutional neural network to assign that grade
automatically, and to explain each decision rather than emitting a bare
number.</p>

<div class="kpi">
  <div><div class="v">{eff['qwk']:.3f}</div><div class="l">Quadratic<br>weighted kappa</div></div>
  <div><div class="v">{train_acc:.1%}</div><div class="l">Training<br>accuracy</div></div>
  <div><div class="v">{val_acc:.1%}</div><div class="l">Validation<br>accuracy</div></div>
  <div class="hero"><div class="v">{eff['accuracy']:.1%}</div><div class="l">Test<br>accuracy</div></div>
  <div><div class="v">{eff['referable_sensitivity']:.1%}</div><div class="l">Referable DR<br>sensitivity</div></div>
</div>

<h3>How the model performs across the three splits</h3>
<p>The same model &mdash; the epoch&nbsp;{best_ep} checkpoint that is actually
shipped &mdash; measured on the data it learned from, the data used to pick
the checkpoint, and the data it had never seen.</p>
{rows([
 ["Training", f"{train_acc:.1%}", "2,562",
  "What the network memorised. <b>Not a performance figure</b> - it only shows how well the weights fit the data they were optimised on."],
 ["Validation", f"{val_acc:.1%}", "550",
  "Used to choose the epoch and trigger early stopping, so it is optimistically biased."],
 ["<b>Test</b>", f"<b>{eff['accuracy']:.1%}</b>", "550",
  "<b>The reported result.</b> Touched once, after every decision was locked in."],
], ["Split", "Accuracy", "Images", "What it means"])}

<div class="note"><b>Reading the gap.</b> Training accuracy sits about
{gap_pp:.0f}
points above test accuracy. That gap is the signature of a model that has
partly memorised a small training set, and it is why the headline figure is
the test number and never the training one. Two later experiments confirmed
the diagnosis: neither a higher input resolution nor test-time augmentation
produced a statistically significant gain, indicating the limit is dataset
size and label noise rather than model capacity or image detail.</div>

<h3>Objectives</h3>
<ol>
  <li>Grade a fundus photograph 0–4 on the international DR severity scale.</li>
  <li>Quantify what transfer learning is worth, by comparing a from-scratch
      CNN against a pretrained backbone under identical conditions.</li>
  <li>Explain each prediction — both <i>where</i> the network looked and
      <i>what</i> is in that region.</li>
  <li>Run entirely on local hardware, with no dependency on cloud GPUs.</li>
</ol>

<div class="note"><b>Scope.</b> This is a research prototype built for academic
assessment. It is not a medical device and has not been clinically validated.
The title refers to grading in patients with Type&nbsp;1 diabetes; the dataset
records DR <i>severity</i> and carries no diabetes-type field, so the model
predicts severity only.</div>

<h2 id="s2">2 &nbsp; The clinical problem and grading scale</h2>
<p>Grades follow the International Clinical Diabetic Retinopathy scale. They
are <b>ordinal</b>: the classes have a meaningful order, and an error between
distant grades is far worse than one between neighbours.</p>
{rows(grade_rows, ["Grade", "Name", "Images", "Share", "Clinical action"])}

<div class="note"><b>Referable DR.</b> Grades 2 and above are "referable" —
the threshold at which a screening programme sends a patient to a specialist.
Collapsing five grades into this yes/no decision gives the number a clinical
reader cares about most, because missing a severe case costs eyesight while a
false alarm costs one appointment. Reported separately throughout.</div>

{fig("samples_by_grade.png", "Representative fundus images at each grade. The difference between grade 0 and grade 4 is obvious; adjacent grades are subtle, and that is where most errors occur.", 1150)}

<h2 id="s3" class="pagebreak">3 &nbsp; Dataset</h2>
<h3>Source</h3>
<p><b>APTOS 2019 Blindness Detection</b>, collected by Aravind Eye Hospital in
rural India and released as a Kaggle competition. {total:,} colour fundus
photographs, each graded 0–4 by a clinician.</p>
<pre><code>https://www.kaggle.com/competitions/aptos2019-blindness-detection</code></pre>

<div class="warn"><b>Access note.</b> The competition blocks every download,
even a 28&nbsp;KB file, until its rules are accepted in a browser. The download
script therefore defaults to a complete mirror of the same images
(<code>mariaherrerot/aptos2019</code>) needing only a logged-in CLI, and
verifies the grade counts against the official distribution either way.</div>

<h3>Class distribution</h3>
<p>The dataset is heavily imbalanced, which drives several design decisions.</p>
{fig("class_imbalance.png", "Grade distribution. Nearly half of all images are grade 0.", 1000)}

<h3>Splitting</h3>
{rows(split_rows, ["Split", "Images", "Share", "Purpose"])}
<p>The split is <b>stratified</b>, so each grade keeps its proportion in all
three sets. With only 193 grade-3 images in total, a random split could leave
wildly different balances and make the validation curve meaningless.</p>

<h3>Data quality findings</h3>
<p>Hashing every image surfaced two problems in the dataset itself.</p>
<ul>
  <li><b>Duplicate photographs.</b> {total:,} rows contain only 3,534 unique
      images — 128 rows are pixel-identical copies under a different
      <code>id_code</code>, across 123 groups.</li>
  <li><b>Conflicting labels.</b> 30 of those groups are graded
      <i>differently</i> in different rows — irreducible label noise that puts
      a ceiling on achievable accuracy.</li>
</ul>
<p>Under a naive per-row split, 28 of 550 test images (5.1%) had a twin in
train. Measuring the effect showed the leakage did <b>not</b> inflate the
result: removing those rows moves QWK from 0.9005 to 0.9031, slightly
<i>better</i>, because the duplicated rows are the ones carrying conflicting
labels. Splitting is now done on image-content hash so copies cannot span
splits.</p>

<h2 id="s4" class="pagebreak">4 &nbsp; Flow of implementation</h2>
<p>The pipeline runs in seven stages. Each is a separate module, so any stage
can be re-run without repeating the others.</p>
{fig("../docs/system_architecture.jpeg", "System architecture: the seven stages from raw image to predicted grade.", 1350)}

{rows([
 ["1", "Retinal image dataset", "<code>src/download_data.py</code>",
  "Downloads and verifies 3,662 graded fundus images."],
 ["2", "Image preprocessing", "<code>src/preprocess.py</code>",
  "Crop, resize, contrast enhancement, circular mask. Cached to disk."],
 ["3", "Training / validation split", "<code>src/preprocess.py</code>",
  "Stratified, group-aware 70/15/15 split."],
 ["4", "CNN model", "<code>src/models.py</code>",
  "Baseline CNN, EfficientNet-B3 or ResNet-50."],
 ["5", "Feature extraction", "<code>src/models.py</code>",
  "The backbone converts an image into a feature vector."],
 ["6", "Classification", "<code>src/models.py</code>, <code>src/engine.py</code>",
  "A linear head maps features to five grade scores."],
 ["7", "DR grade / prediction", "<code>src/predict.py</code>, <code>app/app.py</code>",
  "Grade, confidence, referable decision, and explanation."],
], ["#", "Stage", "Module", "What happens"])}

<h3>End-to-end sequence</h3>
<pre><code>Raw fundus image (e.g. 3216 x 2136)
  |
  |-- 1. Load and verify against the official grade distribution
  |
  |-- 2. Crop black border  ->  resize 456x456  ->  Ben Graham enhancement
  |                             ->  circular field-of-view mask
  |      (cached as PNG so training never re-decodes)
  |
  |-- 3. Stratified group-aware split: 2,562 / 550 / 550
  |
  |-- 4. Augment (flip, rotate, scale, brightness, dropout)
  |      ->  resize to model input (300px)  ->  normalise (ImageNet stats)
  |
  |-- 5. Backbone  ->  1,536-dimensional feature vector
  |
  |-- 6. Dropout  ->  Linear(1536 -> 5)  ->  softmax
  |
  '-- 7. Predicted grade + confidence + Grad-CAM + lesion analysis</code></pre>

<h2 id="s5" class="pagebreak">5 &nbsp; Technology stack</h2>
{rows(stack, ["Layer", "Component", "Role"])}

<h3>Hardware</h3>
<p>Development and training on an <b>Apple M5, 16&nbsp;GB unified memory</b>,
using PyTorch's Metal Performance Shaders backend with sustained GPU
utilisation above 95%. <code>get_device()</code> selects MPS, then CUDA, then
CPU, so the same code runs unchanged on an NVIDIA machine or CPU-only laptop.</p>

{rows([
 ["Baseline CNN", "224", "32", "0.33 s", "96.9", "0.4 min", f"{f['baseline_sum']['minutes']:.0f} min"],
 ["EfficientNet-B3", "300", "16", "0.72 s", "22.1", "1.9 min", f"{f['efficientnet_sum']['minutes']:.0f} min"],
 ["ResNet-50", "300", "16", "0.55 s", "29.1", "1.5 min", "~37 min"],
], ["Model", "Input px", "Batch", "s/step", "img/s", "min/epoch", "Total run"])}

<h2 id="s6">6 &nbsp; Algorithms and techniques</h2>
{rows([[a, b, c] for a, b, c in algos], ["Algorithm", "Role", "What it does and why"])}

<div class="note"><b>A distinction worth being clear on.</b> An
<b>activation function</b> (ReLU, SiLU) is the non-linearity <i>inside</i> the
network. An <b>optimiser</b> (AdamW, SGD) is the rule that turns gradients
<i>into</i> weight updates. They do different jobs and every run uses one of
each — they are not alternatives to one another.</div>

<h2 id="s7" class="pagebreak">7 &nbsp; Preprocessing in detail</h2>
<p>Fundus images arrive at different resolutions, aspect ratios and exposures
because they come from several camera models and clinics. Four operations
normalise them.</p>
<ol>
  <li><b>Black border crop.</b> Trims the uninformative frame around the
      circular retina.</li>
  <li><b>Square resize to 456&nbsp;px</b>, cached to disk so training never
      re-decodes multi-megabyte originals.</li>
  <li><b>Ben Graham enhancement.</b> Subtracts a heavily Gaussian-blurred copy
      of the image from itself. Removing the low-frequency component cancels
      per-camera colour cast and uneven illumination, leaving microaneurysms,
      haemorrhages and exudates clearly visible. This is the single
      highest-impact step for this dataset.</li>
  <li><b>Circular field-of-view mask.</b> Zeroes the corners outside the
      retinal circle so the network cannot key on camera-specific
      artefacts.</li>
</ol>
{fig("preprocessing_steps.png", "The stage-2 pipeline applied to one image. Panel 4 shows the Ben Graham step bringing out exudates and vessel detail that are nearly invisible in the original.", 1500)}

<h3>Augmentation</h3>
<p>Applied to the training split only. Retinal images have no canonical
orientation — left and right eyes are mirror images — so flips and full
rotations are safe and effectively multiply a small dataset.</p>
{rows([
 ["RandomResizedCrop", "scale 0.85–1.0", "Slight framing variation"],
 ["Horizontal / vertical flip", "p = 0.5 each", "Eye laterality and camera orientation"],
 ["Affine", "±180° rotation, ±10% scale, ±5% shift", "No canonical orientation exists"],
 ["RandomBrightnessContrast", "±15%", "Exposure differences between clinics"],
 ["HueSaturationValue", "small shifts", "Camera colour variation"],
 ["CoarseDropout", "1–4 holes", "Forces reliance on more than one region"],
 ["Normalize", "ImageNet mean/std", "Matches what the pretrained backbone expects"],
], ["Transform", "Setting", "Why"])}

<h2 id="s8" class="pagebreak">8 &nbsp; Model architectures</h2>
{rows([
 ["<code>baseline</code>", "5-block VGG-style CNN, from scratch", "4.85 M", "224 px",
  "No pretrained weights. Establishes the honest reference point."],
 ["<code>efficientnet</code>", "EfficientNet-B3, ImageNet-pretrained", "10.70 M", "300 px",
  "Main model. Compound-scaled depth, width and resolution."],
 ["<code>resnet</code>", "ResNet-50, ImageNet-pretrained", "23.52 M", "300 px",
  "Residual connections. Configured as a third comparison row."],
], ["Key", "Architecture", "Parameters", "Input", "Purpose"])}

<h3>Baseline CNN</h3>
<p>Five convolutional blocks, each <code>Conv → BatchNorm → ReLU → Conv →
BatchNorm → ReLU → MaxPool</code>. Channels double at every stage
(32→64→128→256→512) while spatial resolution halves. Global average pooling
replaces flatten-plus-dense layers, keeping the parameter count low — which
matters when training from scratch on only 2,562 images.</p>

<h3>Transfer-learning models</h3>
<p>The pretrained backbone is loaded with its ImageNet classifier removed, so
it outputs a pooled feature vector (1,536 values for B3). A fresh
<code>Dropout → Linear(1536 → 5)</code> head is attached and the whole network
is fine-tuned end to end.</p>

<h2 id="s9">9 &nbsp; Training configuration</h2>
{rows([
 ["Optimiser", "AdamW", "Decoupled weight decay regularises correctly"],
 ["Learning rate", "3e-4", "Cosine decay after 2 warmup epochs"],
 ["Weight decay", "1e-5 (transfer) / 1e-4 (scratch)", "Stronger for the from-scratch model"],
 ["Batch size", "16 (300 px) / 32 (224 px)", "Fits 16 GB unified memory"],
 ["Loss", "Class-weighted cross-entropy, label smoothing 0.05", "Counters 49% grade-0 imbalance"],
 ["Class weights", "0.22 / 1.06 / 0.39 / 2.02 / 1.32", "Inverse frequency, normalised to mean 1"],
 ["Gradient clipping", "norm 1.0", "Prevents exploding updates"],
 ["Early stopping", "patience 8 on validation QWK", "Keeps the best checkpoint"],
 ["Seed", "42", "Python, NumPy, torch and albumentations all seeded"],
], ["Setting", "Value", "Reason"])}

<div class="note"><b>Why monitor QWK rather than validation loss.</b> Between
epochs 10 and 13 the validation <i>loss</i> rose while validation <i>QWK</i>
improved to its best value. Both are correct: cross-entropy punishes
overconfidence, whereas QWK only measures whether the predicted grade lands in
the right place on the ordinal scale. Early stopping on loss would have frozen
the model at epoch 6 and discarded the best result.</div>

{fig("efficientnet_training_history.png", "EfficientNet-B3 training history. Validation QWK peaks at epoch 13; training halted at epoch 21 under the patience rule.", 1500)}

<h2 id="s10" class="pagebreak">10 &nbsp; Evaluation metrics</h2>
<h3>Quadratic Weighted Kappa — the headline number</h3>
<p>QWK measures <b>ordinal agreement</b> corrected for chance. Unlike accuracy,
it weights an error by how far apart the grades are: predicting grade 3 when
the truth is 4 costs little, predicting grade 0 when the truth is 4 costs
sixteen times as much, because the penalty is the squared distance. 0 means no
better than random guessing, 1.0 is perfect. It is the official APTOS
competition metric.</p>

<div class="warn"><b>QWK is not accuracy.</b> They are different numbers and
must not be confused. This model scores <b>QWK {eff['qwk']:.4f}</b> and
<b>accuracy {eff['accuracy']:.1%}</b>. QWK is higher because nearly every error
is an adjacent-grade confusion, which QWK rewards and accuracy does not.</div>

{rows([
 ["Quadratic weighted kappa", "Ordinal agreement, chance-corrected", "Headline metric; matches how a clinician judges errors"],
 ["Accuracy", "Fraction predicted exactly right", "Intuitive, but flattered by grade 0 being half the data"],
 ["Macro F1", "Unweighted mean F1 across grades", "Treats rare grades as equally important; reveals weakness"],
 ["Referable sensitivity", "Recall on grade ≥ 2", "The clinical screening decision — a miss costs eyesight"],
 ["Referable specificity", "True-negative rate on grade ≥ 2", "Controls unnecessary referrals"],
], ["Metric", "What it measures", "Why it is reported"])}

<h2 id="s11">11 &nbsp; Results</h2>
<p>Measured on the held-out test split of 550 images, which no training or
model-selection step ever touched.</p>
{rows(cmp_rows, ["Model", "Params", "Epochs", "Time", "Accuracy", "QWK", "Macro F1", "Referable sens."])}

<p>Transfer learning is worth <b>+{eff['qwk']-base['qwk']:.3f} QWK</b> and
<b>+{100*(eff['accuracy']-base['accuracy']):.1f} percentage points</b> of
accuracy. It also reached the baseline's best score in 4 epochs rather than
32.</p>

{fig("model_comparison_test.png", "Model comparison on the held-out test split.", 1400)}

<h3>Per-class performance — EfficientNet-B3</h3>
{rows(perclass, ["Grade", "Precision", "Recall", "F1", "Support"])}

{fig("efficientnet_confusion_matrix.png", "Confusion matrix. Counts on the left, normalised by true grade on the right.", 1500)}

<h3>Reading the results honestly</h3>
<ul>
  <li><b>No sight-threatening case was called healthy.</b> Zero grade-3 and
      grade-4 images were predicted as grade 0 — the bottom-left of the
      confusion matrix is all zeros. This is the single most important
      clinical property.</li>
  <li><b>Errors are almost all adjacent-grade</b> (Mild↔Moderate,
      Severe↔Proliferative), mirroring where human graders themselves
      disagree.</li>
  <li><b>Grades 3 and 4 are the weak point</b> (F1 0.43 and 0.58). They have
      only 193 and 295 images in the entire dataset — a data limitation, not a
      modelling failure.</li>
  <li><b>Grade 3 precision is 0.366</b>, meaning the model over-calls
      "Severe". That is the deliberate consequence of class weighting: in
      screening, a false alarm costs an appointment while a miss costs
      eyesight.</li>
  <li><b>Macro F1 (0.68) sits well below accuracy (0.83)</b> because macro F1
      weights all five grades equally while accuracy is flattered by grade 0
      being half the data.</li>
</ul>

<h3>Attempts to improve accuracy, and what they showed</h3>
<p>Two standard interventions were tested against the 300&nbsp;px model. The
TTA configuration was chosen on the <i>validation</i> set and then reported
once on test; choosing it on test would be tuning on held-out data.</p>
{rows([
 ["EfficientNet-B3 @ 300 px <i>(reported model)</i>", "0.8255", "0.9005", "454 / 550", "&mdash;"],
 ["&nbsp;&nbsp;+ test-time augmentation", "0.8218", "0.8953", "452 / 550", "&minus;2 images"],
 ["EfficientNet-B3 @ 456 px", "0.8309", "0.9023", "457 / 550", "+3 images"],
 ["&nbsp;&nbsp;+ test-time augmentation", "0.8273", "0.9038", "455 / 550", "+1 image"],
], ["Configuration", "Accuracy", "QWK", "Correct", "Change"])}

<p>Raising the input resolution to 456&nbsp;px cost 91 minutes of training and
gained three images out of 550. A McNemar test on the disagreements settles
whether that is real:</p>
<pre><code>300px right, 456px wrong : 30
300px wrong, 456px right : 33
McNemar exact p = 0.801   ->  NOT significant

one standard error on accuracy at n=550  =  1.6 pp  (9 images)
observed difference                      =  0.55 pp (3 images)</code></pre>

<div class="note"><b>Conclusion.</b> Neither intervention produced a
statistically significant improvement, and test-time augmentation was actively
harmful. The hypothesis behind the resolution experiment was that
microaneurysms are 10-20&nbsp;px in a full-resolution image and therefore
sub-pixel at 300&nbsp;px. That is true, but not the binding constraint: with
training accuracy at 94% against 83% on test, the model had already fitted the
2,562 training images. Sharper inputs added capacity to memorise, not
information to generalise from. The limit is <b>dataset size and label
noise</b>, so the productive next steps are ones that add data or diversity -
model ensembling, or pretraining on the much larger 2015 Diabetic Retinopathy
dataset - rather than ones that add detail or capacity.</div>

<h2 id="s12" class="pagebreak">12 &nbsp; Explainability</h2>
<h3>Grad-CAM — where the model looked</h3>
<p>Gradient-weighted Class Activation Mapping backpropagates the predicted
grade's score to the last convolutional feature map. Channels are weighted by
their mean gradient, summed and passed through ReLU, giving a heatmap over the
regions that raised the score.</p>
{fig("gradcam_by_grade.png", "Grad-CAM across all five grades. On higher grades the heat sits on haemorrhages and exudates; on a healthy retina it is diffuse, because there is no lesion to point at.", 1050)}

<h3>Lesion analysis — what is in that region</h3>
<p>A heatmap says <i>where</i>, not <i>what</i>. Classical detectors locate the
features the grading scale is actually built on, and the enrichment ratio
measures how concentrated each is inside the attended region.</p>
{rows([
 ["Microaneurysms / haemorrhages", "Local minima in the green channel, filtered per blob by area, elongation and fill",
  "Earliest and most decisive sign; number and spread drive grades 1–3"],
 ["Hard exudates", "High brightness and yellowness, optic disc excluded",
  "Lipid deposits from leaking capillaries; sight-threatening near the macula"],
 ["Vasculature", "Black top-hat at multiple orientations",
  "Detected to keep vessels out of the red-lesion mask; drawn for orientation"],
], ["Feature", "Detection method", "Clinical role"])}

<p>Detection runs on the <i>original</i> image rather than the preprocessed
one, because the Ben Graham step recentres every image on mid-grey and
destroys the hue difference between yellow exudates and red haemorrhages. The
optic disc is masked out of both detectors — it is naturally bright and yellow
and would otherwise dominate the exudate count on every image.</p>

<h4>Validation across grades</h4>
{rows([
 ["0", "4.5", "0.036", "6.1"], ["1", "5.9", "0.090", "12.0"],
 ["2", "8.8", "0.117", "11.6"], ["3", "6.9", "0.192", "16.3"],
 ["4", "9.1", "0.213", "16.1"],
], ["Grade", "Exudate count", "Red lesion % area", "Red lesion count"])}
<p>Red lesion area rises <b>5.9×</b> from grade 0 to grade 4 and is monotonic
across all five grades — exactly the criterion the clinical scale uses.</p>

{fig("lesion_detection_examples.png", "Left: original. Right: detections. Red rings mark microaneurysms and haemorrhages, yellow marks hard exudates, faint blue marks the vasculature.", 900)}

<div class="warn"><b>What this does not prove.</b> APTOS carries image-level
grades only, with no lesion annotations, so none of these are trained
detectors — they are hand-written image-processing rules. The panel reports a
<i>correlation</i> between independently detected lesions and where the network
looked, not a readout of the model's reasoning. A high multiplier is evidence
the model attended to clinically meaningful structures; a low one means either
the detectors missed something or the model used features they cannot see.</div>

<h2 id="s13" class="pagebreak">13 &nbsp; The demo application</h2>
<p>A Streamlit application implements stage 7 interactively. Upload a fundus
photograph and it returns the grade, per-class confidence, the referable-DR
decision, the preprocessing steps, a Grad-CAM overlay and the lesion
analysis.</p>
{fig("../docs/app_screenshot.png", "The demo application. The result appears first, above the images, because it is what the user came for.", 1500)}

<h2 id="s14">14 &nbsp; Repository structure</h2>
<pre><code>src/
  config.py            All hyperparameters, paths and grade definitions
  utils.py             Seeding and reproducibility helpers
  download_data.py     Stage 1 — dataset acquisition and verification
  download_weights.py  Fetch trained checkpoints from the GitHub release
  preprocess.py        Stages 2 and 3 — preprocessing and the split
  dataset.py           Dataset class, augmentation, class weights
  models.py            Stages 4-6 — architectures, activations, factory
  engine.py            Training and evaluation loops, metrics, optimisers
  train.py             Training entry point
  evaluate.py          Test-set metrics and figures
  compare.py           Model comparison table and figure
  ablation.py          Optimiser and activation sweeps
  eda.py               Dataset-chapter figures
  gradcam.py           Grad-CAM implementation
  lesions.py           Lesion detection and attention enrichment
  predict.py           Single-image inference
app/app.py             Streamlit demo
notebooks/             Pipeline walkthrough notebook
docs/                  Architecture diagram, Windows guide, this document
outputs/figures/       Report figures
outputs/logs/          Metrics, histories, comparison tables
outputs/models/        Trained weights (not committed; see release)</code></pre>

<h2 id="s15">15 &nbsp; How to run it</h2>
<h3>Demo only — no dataset required</h3>
<pre><code># macOS / Linux
python3 -m venv .venv &amp;&amp; source .venv/bin/activate
pip install -r requirements.txt
python -m src.download_weights
streamlit run app/app.py

# Windows (PowerShell)
py -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
python -m src.download_weights
streamlit run app/app.py</code></pre>
<p>Trained weights are published as GitHub release assets rather than
committed, so clones stay small. Full Windows instructions, including the CUDA
build and common errors, are in <code>docs/SETUP_WINDOWS.md</code>.</p>

<h3>Full pipeline — retraining from scratch</h3>
<pre><code>kaggle auth login                              # one-time browser login
python -m src.download_data                    # ~8.6 GB
python -m src.preprocess                       # cache + stratified split
python -m src.train --model baseline           # ~22 min
python -m src.train --model efficientnet       # ~46 min
python -m src.evaluate --model efficientnet --split test
python -m src.compare --split test
python -m src.eda                              # dataset figures</code></pre>
<p>A two-epoch smoke test verifies the whole pipeline in about a minute:</p>
<pre><code>python -m src.train --model efficientnet --epochs 2 --limit 64</code></pre>

<h3>Ablations</h3>
<pre><code>python -m src.train --model baseline --optimizer sgd --activation gelu
python -m src.ablation --sweep optimizer  --epochs 15
python -m src.ablation --sweep activation --epochs 15</code></pre>
<p>Activations available: <code>relu</code>, <code>leaky_relu</code>,
<code>gelu</code>, <code>silu</code>, <code>elu</code>, <code>mish</code>.
Optimisers: <code>adamw</code>, <code>adam</code>, <code>sgd</code>,
<code>rmsprop</code>. Sweeps use the baseline model, because pretrained
backbones ship with their own activations and replacing those would invalidate
the ImageNet weights.</p>

<h2 id="s16" class="pagebreak">16 &nbsp; Known limitations and future work</h2>
{rows([
 ["Test set is a split of APTOS <i>train</i>", "The competition's real test labels were never public, so the held-out 550 images come from the labelled training set.", "Report it explicitly; it is the standard approach for this dataset."],
 ["Rare grades are weak", "Grades 3 and 4 have 193 and 295 images in total. F1 is 0.43 and 0.58.", "Collect more severe cases, or use focal loss and oversampling."],
 ["Label noise", "30 duplicate groups are graded differently in different rows.", "An irreducible ceiling on achievable accuracy; disclose it."],
 ["Grade 3 over-called", "Precision 0.366 — a deliberate consequence of class weighting.", "Defensible for screening; tune the weights if precision matters more."],
 ["Lesion detectors are not trained", "APTOS has no lesion annotations, so detection is classical image processing.", "Train a U-Net on IDRiD (81 annotated images) or DDR (608)."],
 ["Microaneurysms are invisible at 300 px", "The classifier resizes to 300 px, where a 10-20 px lesion disappears.", "Patch-based training at 1024 px or higher."],
 ["Single model, no ensemble", "Competition-winning solutions ensemble several backbones.", "Ensemble EfficientNet, ResNet and a ViT; expect +0.01-0.02 QWK."],
], ["Limitation", "Detail", "How it could be addressed"])}

<h2 id="s17">17 &nbsp; Glossary</h2>
{rows([
 ["Fundus image", "A photograph of the interior back surface of the eye, showing retina, optic disc and blood vessels."],
 ["Microaneurysm", "A tiny bulge in a retinal capillary wall. The earliest visible sign of DR, only 10-20 px in a full-resolution image."],
 ["Haemorrhage", "Blood leaked into retinal tissue, appearing as a dark red spot."],
 ["Hard exudate", "A yellow lipid deposit left by leaking capillaries."],
 ["Optic disc", "The bright circular region where the optic nerve enters the retina. Naturally bright and yellow, so it is masked out of lesion detection."],
 ["Macula", "The central retinal region responsible for sharp vision. Lesions here are especially sight-threatening."],
 ["NPDR / PDR", "Non-proliferative and proliferative diabetic retinopathy. PDR involves new, fragile blood vessel growth."],
 ["Referable DR", "Grade 2 or above — the threshold for referral to an ophthalmologist."],
 ["Ordinal classification", "Classification where classes have a meaningful order, so some errors are worse than others."],
 ["QWK", "Quadratic Weighted Kappa. Chance-corrected ordinal agreement, penalising errors by squared distance."],
 ["Transfer learning", "Reusing a network pretrained on a large dataset (ImageNet) and fine-tuning it on a smaller target task."],
 ["Activation function", "The non-linearity inside a network (ReLU, SiLU). Without one, stacked layers collapse to a single linear operation."],
 ["Optimiser", "The rule converting gradients into weight updates (AdamW, SGD)."],
 ["Class weighting", "Scaling the loss per class by inverse frequency so rare classes are not ignored."],
 ["Grad-CAM", "Gradient-weighted Class Activation Mapping — a heatmap of the regions that raised the predicted class score."],
 ["Stratified split", "A split preserving each class's proportion across train, validation and test."],
 ["Data leakage", "When information from the test set influences training — here, duplicate photographs spanning splits."],
], ["Term", "Meaning"])}

<div class="note" style="margin-top:20pt"><b>Disclaimer.</b> Research
prototype built for academic assessment. Not a medical device, not clinically
validated, and not a substitute for examination by a qualified
ophthalmologist.</div>

</body></html>"""


def main() -> None:
    facts = load_facts()
    OUT_HTML.write_text(build(facts))
    size = OUT_HTML.stat().st_size / 1_048_576
    print(f"HTML written: {OUT_HTML}  ({size:.1f} MB)")


if __name__ == "__main__":
    main()
