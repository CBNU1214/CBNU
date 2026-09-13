"""
결함 커버리지 곡선, 회로 규모별 테스트 난이도, 미검출(Redundant 후보) 결함 분석
"""
import json
import os
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from netlist import parse_verilog

matplotlib.use("Agg")
warnings.filterwarnings("ignore")
from matplotlib import font_manager
_available_fonts = {f.name for f in font_manager.fontManager.ttflist}
_plot_font = next((f for f in ["AppleGothic", "Malgun Gothic", "Noto Sans CJK KR", "NanumGothic"] if f in _available_fonts), "DejaVu Sans")
sns.set_theme(style="whitegrid", font=_plot_font)
plt.rcParams["axes.unicode_minus"] = False

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

with open("data/circuit_summary.json", encoding="utf-8") as f:
    summary = json.load(f)
circuits_info = pd.DataFrame(summary["circuits"])
faults_df = pd.read_csv("data/fault_results.csv")
CIRCUITS = circuits_info.circuit.tolist()

print(circuits_info[["circuit", "n_gates", "n_faults", "final_coverage_1000vec", "n_undetected"]])

# ------------------------------------------------------------------
# 1. 결함 커버리지 곡선 (벡터 수에 따른 누적 커버리지)
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5.5))
palette = sns.color_palette("husl", len(CIRCUITS))
for row, color in zip(summary["circuits"], palette):
    curve = row["coverage_curve"]
    ax.plot(range(1, len(curve) + 1), np.array(curve) * 100, label=f"{row['circuit']} ({row['n_gates']}게이트)",
            color=color, linewidth=1.6)
ax.set_xscale("log")
ax.set_xlabel("테스트 벡터 수 (로그 스케일)")
ax.set_ylabel("결함 커버리지 (%)")
ax.set_title("랜덤 테스트 벡터 수에 따른 결함 커버리지 — 회로별 검출 특성 비교")
ax.legend(fontsize=9)
ax.axhline(100, color="gray", linewidth=0.5, linestyle=":")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/01_coverage_curves.png", dpi=150)
plt.close()
print("[1] 커버리지 곡선 시각화 완료")

# ------------------------------------------------------------------
# 2. c17 — 랜덤 vs Exhaustive 비교
# ------------------------------------------------------------------
c17_random = next(r for r in summary["circuits"] if r["circuit"] == "c17")["coverage_curve"][:32]
c17_exh = summary["c17_exhaustive_coverage"]
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(range(1, 33), np.array(c17_random) * 100, marker="o", markersize=3, label="랜덤 벡터", color="#c0392b")
ax.plot(range(1, 33), np.array(c17_exh) * 100, marker="s", markersize=3, label="Exhaustive(전수 32개)", color="#005596")
ax.set_xlabel("사용한 벡터 수")
ax.set_ylabel("결함 커버리지 (%)")
ax.set_title("c17 — 랜덤 벡터 vs 전수(Exhaustive) 테스트 비교\n(입력 5개, 가능한 조합 32개뿐이라 전수 테스트 가능)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/02_c17_random_vs_exhaustive.png", dpi=150)
plt.close()
print("[2] c17 랜덤 vs 전수 비교 시각화 완료")

# ------------------------------------------------------------------
# 3. 회로 규모 vs 테스트 난이도
# ------------------------------------------------------------------
def vectors_to_reach(curve, target):
    arr = np.array(curve)
    idx = np.flatnonzero(arr >= target)
    return int(idx[0] + 1) if len(idx) else None

circuits_info["vec_to_95pct"] = [vectors_to_reach(r["coverage_curve"], 0.95) for r in summary["circuits"]]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
axes[0].scatter(circuits_info.n_gates, circuits_info.n_faults, s=60, color="#005596")
for _, r in circuits_info.iterrows():
    axes[0].annotate(r.circuit, (r.n_gates, r.n_faults), fontsize=8, xytext=(4, 4), textcoords="offset points")
axes[0].set_xlabel("게이트 수"); axes[0].set_ylabel("체크포인트 결함 수")
axes[0].set_title("회로 규모 vs 결함 수")

axes[1].bar(circuits_info.circuit, circuits_info.vec_to_95pct, color="#8e44ad")
axes[1].set_title("95% 커버리지 도달까지 필요한 벡터 수")
axes[1].set_ylabel("벡터 수")

axes[2].bar(circuits_info.circuit, circuits_info.n_undetected, color="#c0392b")
axes[2].set_title("1000벡터 후에도 미검출된 결함 수\n(Redundant Fault 후보)")
axes[2].set_ylabel("결함 수")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/03_circuit_scale_vs_testability.png", dpi=150)
plt.close()
print("[3] 회로 규모 vs 테스트 난이도 시각화 완료")

# ------------------------------------------------------------------
# 4. 미검출(Redundant 후보) 결함의 게이트 타입 breakdown
# ------------------------------------------------------------------
gate_type_by_net = {}
for name in CIRCUITS:
    c = parse_verilog(f"data/{name}.v")
    for g in c["gates"]:
        gate_type_by_net[(name, g["out"])] = g["type"]
    for inp in c["inputs"]:
        gate_type_by_net[(name, inp)] = "primary_input"

undetected = faults_df[~faults_df.detected].copy()
undetected["gate_type"] = undetected.apply(lambda r: gate_type_by_net.get((r.circuit, r.fault_net), "unknown"), axis=1)

fig, ax = plt.subplots(figsize=(7, 4.5))
undetected.gate_type.value_counts().plot(kind="bar", ax=ax, color="#c0392b")
ax.set_title(f"미검출 결함 {len(undetected)}건의 게이트 타입 분포")
ax.set_ylabel("결함 수")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/04_undetected_by_gate_type.png", dpi=150)
plt.close()
print("[4] 미검출 결함 게이트 타입 분포 시각화 완료")
print(undetected.groupby(["circuit", "gate_type"]).size())

# ------------------------------------------------------------------
# 5. summary.json
# ------------------------------------------------------------------
out_summary = {
    "circuits": circuits_info[["circuit", "n_gates", "n_inputs", "n_outputs", "n_faults",
                                "final_coverage_1000vec", "n_undetected", "vec_to_95pct"]].to_dict("records"),
    "total_faults_simulated": int(faults_df.shape[0]),
    "n_vectors_per_circuit": 1000,
    "undetected_gate_type_breakdown": undetected.gate_type.value_counts().to_dict(),
    "c17_fully_testable": bool(next(r for r in summary["circuits"] if r["circuit"] == "c17")["n_undetected"] == 0),
}
with open("summary.json", "w", encoding="utf-8") as f:
    json.dump(out_summary, f, ensure_ascii=False, indent=2)

print("\n분석 완료. figures/ 폴더와 summary.json을 확인하세요.")
