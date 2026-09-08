"""Build the Grad-CAM figure: one row per DR grade.

Run:  python -m src.make_gradcam_figure
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, pandas as pd, numpy as np, cv2
from src import config as C
from src.predict import predict_file

SURFACE="#fcfcfb"; INK="#0b0b0b"; INK2="#52514e"

def main() -> None:
    splits = pd.read_csv(C.SPLITS_CSV); test = splits[splits.split=="test"]
    rng = np.random.default_rng(7)

    def pad_square(img):
        """Letterbox to a square so every panel is the same size and the column
        headers line up. The image itself is untouched, only padded."""
        h, w = img.shape[:2]
        side = max(h, w)
        out = np.zeros((side, side, 3), img.dtype)
        y0, x0 = (side - h) // 2, (side - w) // 2
        out[y0:y0+h, x0:x0+w] = img
        return out

    rows=[]
    for g in range(5):
        pool = test[test.label==g]
        # prefer a correctly-classified example so the heatmap is interpretable
        chosen=None
        for iid in rng.permutation(pool.id_code.values)[:6]:
            r = predict_file(C.IMAGES_RAW/f"{iid}.png", model_key="efficientnet")
            if chosen is None: chosen=(iid,r)
            if r["grade"]==g: chosen=(iid,r); break
        rows.append((g,)+chosen)

    fig, axes = plt.subplots(5, 3, figsize=(11, 18.5), facecolor=SURFACE)
    for i,(g,iid,r) in enumerate(rows):
        raw = pad_square(cv2.cvtColor(cv2.imread(str(C.IMAGES_RAW/f"{iid}.png")), cv2.COLOR_BGR2RGB))
        for j,(img,t) in enumerate([(raw,"Original"),(r["processed"],"Preprocessed"),(r["overlay"],"Grad-CAM")]):
            ax=axes[i,j]; ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values(): s.set_visible(False)
            if i==0: ax.set_title(t, fontsize=12, color=INK, fontweight="bold")
        ok = "correct" if r["grade"]==g else f"predicted {C.CLASS_SHORT[r['grade']]}"
        axes[i,0].set_ylabel(f"True: {C.CLASS_SHORT[g]}\n{ok} ({r['confidence']:.0%})",
                             fontsize=10.5, color=INK, rotation=0, ha="right", va="center", labelpad=58)
    fig.suptitle("Grad-CAM explanations across DR grades (EfficientNet-B3)",
                 fontsize=15, fontweight="bold", color=INK, y=0.997)
    fig.text(0.5, 0.005, "Warmer regions increased the score for the predicted grade.",
             ha="center", fontsize=10, color=INK2)
    fig.tight_layout(rect=[0,0.012,1,0.985])
    out = C.FIGURE_DIR/"gradcam_by_grade.png"
    fig.savefig(out, dpi=125, bbox_inches="tight", facecolor=SURFACE); plt.close(fig)
    print("figure ->", out)


if __name__ == "__main__":
    main()
