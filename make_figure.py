"""Regenerates subject_size.pdf (Fig. 1) from the saved per-seed results.
Input: results/ALL_RESULTS.xlsx, sheet accuracy_by_size, from
https://github.com/danishali-codes/breaking-camouflage-cod10k
Usage: python make_figure.py path/to/ALL_RESULTS.xlsx subject_size.pdf
"""
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src, out = sys.argv[1], sys.argv[2]
s = pd.read_excel(src, sheet_name="accuracy_by_size")
g = (100 * s.groupby(["family", "size", "res", "bin"])["acc"].agg(["mean", "std"])).reset_index()
fams = [("convnext", "ConvNeXt"), ("dinov2", "DINOv2"), ("vit", "ViT"), ("clip", "CLIP")]
stages = [("B", 224, "Base, 224 px", "#2678b2", "o", "-"),
          ("B", 448, "Base, 448 px", "#e67824", "s", "--"),
          ("L", 224, "Large, 224 px", "#249c70", "^", ":")]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42})
fig, axes = plt.subplots(2, 2, figsize=(5.2, 3.3), sharex=True, sharey=True)
for ax, (fam, title) in zip(axes.ravel(), fams):
    for size, res, lab, col, mk, ls in stages:
        d = g[(g.family == fam) & (g["size"] == size) & (g.res == res)].sort_values("bin")
        ax.errorbar(d["bin"], d["mean"], yerr=d["std"], color=col, marker=mk, ms=3.5, lw=1.2, ls=ls,
                    capsize=2.5, elinewidth=0.9, label=lab)
    ax.set_title(title, loc="left", fontsize=10)
    ax.set_xticks(range(5)); ax.set_xticklabels([f"Q{i}" for i in range(1, 6)])
    ax.set_ylim(40, 100); ax.set_yticks(range(40, 101, 10))
    ax.grid(axis="y", color="#dbe0e6", lw=0.7); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
fig.supylabel("Top-1 accuracy (%)", x=0.02, fontsize=9)
fig.supxlabel("Subject-size quintile (smallest to largest)", y=0.085, fontsize=9)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, handlelength=3)
fig.tight_layout(rect=(0.03, 0.11, 1, 1))
fig.savefig(out)
