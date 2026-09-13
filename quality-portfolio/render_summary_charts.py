"""Create reviewable SVG summaries from the committed result JSON files."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})

pcb = ROOT / "pcb-defects/results"
data = json.loads((pcb / "reproduced_metrics.json").read_text())
cm = np.array(data["SVM"]["confusion_matrix"])
labels = ["Missing hole", "Mouse bite", "Open circuit", "Short", "Spur", "Spurious copper"]
fig, ax = plt.subplots(figsize=(9, 7), layout="constrained")
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(6), labels, rotation=35, ha="right")
ax.set_yticks(range(6), labels)
for i in range(6):
    for j in range(6):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > 65 else "#17324d")
ax.set(xlabel="Predicted defect type", ylabel="True defect type",
       title="PCB defect classification | SVM, 739 evaluation instances")
fig.colorbar(im, ax=ax, label="Instances", shrink=.8)
fig.savefig(pcb / "confusion_matrix.svg")
plt.close(fig)

circuit = ROOT / "circuit-faults/results"
rows = json.loads((circuit / "reproduced_metrics.json").read_text())["circuits"]
fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
values = [r["final_coverage_1000vec"] * 100 for r in rows]
bars = ax.barh([r["circuit"] for r in rows], values, color="#236b8e")
ax.bar_label(bars, labels=[f"{v:.2f}%" for v in values], padding=5)
ax.set(xlim=(0, 112), xlabel="Detected faults / defined fault set (%)",
       title="ISCAS85 | 1,000 random vectors per circuit")
ax.invert_yaxis()
ax.spines[["top", "right"]].set_visible(False)
fig.savefig(circuit / "coverage.svg")
plt.close(fig)

manufacturing = ROOT / "manufacturing-defects/results"
data = json.loads((manufacturing / "source_summary.json").read_text())["analyze"]
values = [data["low_group_actual_defect_rate"]*100,
          data["high_group_actual_defect_rate"]*100]
fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
bars = ax.bar(["MaintenanceHours <= 12\nn = 1,751", "MaintenanceHours > 12\nn = 1,489"],
              values, color=["#236b8e", "#c16b42"], width=.5)
ax.bar_label(bars, labels=[f"{v:.2f}%" for v in values], padding=5)
ax.set(ylim=(0, 110), ylabel="DefectStatus = 1 in dataset (%)",
       title="Manufacturing simulation data | observed groups")
ax.text(.5, -.19, "Association only: not a measured intervention effect.",
        transform=ax.transAxes, ha="center", fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
fig.savefig(manufacturing / "group_comparison.svg", bbox_inches="tight")
plt.close(fig)
