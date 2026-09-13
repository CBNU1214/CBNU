"""
제조공정 불량 개선 (DMAIC) 분석
- 데이터: Kaggle "Predicting Manufacturing Defects Dataset" (rabieelkharoua), 3,240행 17개 변수
- 주의: 특정 산업(PCB/방산 등)에 종속된 데이터가 아닌 범용 제조공정 시뮬레이션 데이터
- 목표: 불량 원인분석에 그치지 않고 DMAIC(정의-측정-분석-개선-관리) 흐름으로
  "조치 후 불량률이 얼마나 개선되는가"까지 정량화 (Define/Measure -> Analyze -> Improve -> Control)
"""
import json
import os
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")
warnings.filterwarnings("ignore")
from matplotlib import font_manager
_available_fonts = {f.name for f in font_manager.fontManager.ttflist}
_plot_font = next((f for f in ["AppleGothic", "Malgun Gothic", "Noto Sans CJK KR", "NanumGothic"] if f in _available_fonts), "DejaVu Sans")
sns.set_theme(style="whitegrid", font=_plot_font)
plt.rcParams["axes.unicode_minus"] = False

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

# ------------------------------------------------------------------
# 1. 데이터 로드 (Define)
# ------------------------------------------------------------------
df = pd.read_csv("data/manufacturing_defect_dataset.csv")
n_total = len(df)
baseline_rate = df.DefectStatus.mean()
print(f"[1] 데이터 {n_total:,}행 로드. 기준(Baseline) 불량률: {baseline_rate:.4f}")

# ------------------------------------------------------------------
# 2. Measure — SPC p-chart (행 순서를 가상의 생산 로트 순서로 간주)
# ------------------------------------------------------------------
LOT_SIZE = 50
df["lot"] = df.index // LOT_SIZE
lot_stats = df.groupby("lot")["DefectStatus"].agg(["mean", "count"])
lot_stats = lot_stats[lot_stats["count"] == LOT_SIZE]

pbar = baseline_rate
ucl = pbar + 3 * np.sqrt(pbar * (1 - pbar) / LOT_SIZE)
lcl = max(0.0, pbar - 3 * np.sqrt(pbar * (1 - pbar) / LOT_SIZE))
oc_lots = lot_stats[(lot_stats["mean"] > ucl) | (lot_stats["mean"] < lcl)]
print(f"[2] p-chart: p̄={pbar:.3f}, UCL={ucl:.3f}, LCL={lcl:.3f}, "
      f"로트 {len(lot_stats)}개 중 관리이탈 {len(oc_lots)}개")

plt.figure(figsize=(10, 4.5))
plt.plot(lot_stats.index, lot_stats["mean"], marker="o", ms=4, lw=1, color="#3b6ea5", label="로트별 불량률")
plt.axhline(pbar, color="#2e7d32", ls="-", lw=1.2, label=f"p̄ = {pbar:.3f}")
plt.axhline(ucl, color="#c0392b", ls="--", lw=1.2, label=f"UCL = {ucl:.3f}")
plt.axhline(lcl, color="#c0392b", ls="--", lw=1.2, label=f"LCL = {lcl:.3f}")
plt.scatter(oc_lots.index, oc_lots["mean"], color="#c0392b", zorder=5, s=45, label="관리이탈 로트")
plt.title(f"불량률 p-관리도 (로트 크기 n={LOT_SIZE}, 총 {len(lot_stats)}개 로트)")
plt.xlabel("로트 번호 (가상의 생산 순서)")
plt.ylabel("로트별 불량률")
plt.legend(loc="lower left", fontsize=9)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/01_control_chart.png", dpi=150)
plt.close()

# ------------------------------------------------------------------
# 3. Analyze — 상관분석 + RandomForest 변수중요도
# ------------------------------------------------------------------
feat_cols = [c for c in df.columns if c not in ("DefectStatus", "DefectRate", "lot")]
corr_target = df[feat_cols + ["DefectStatus", "DefectRate"]].corr()
print("[3] DefectRate와의 선형 상관계수 (절대값 기준 상위 3개):")
print(corr_target["DefectRate"].drop(["DefectRate", "DefectStatus"]).abs().sort_values(ascending=False).head(3))

plt.figure(figsize=(9, 7.5))
sns.heatmap(corr_target, cmap="RdBu_r", center=0, annot=False, cbar_kws={"shrink": 0.8})
plt.title("변수 간 상관관계 히트맵")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/02_correlation_heatmap.png", dpi=150)
plt.close()

X = df[feat_cols]
y = df["DefectStatus"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
rf = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1, class_weight="balanced")
rf.fit(X_train, y_train)
auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
print(f"[3] RandomForest DefectStatus 분류 AUC={auc:.4f} "
      f"(단순정확도는 84% 다수클래스 베이스라인과 비슷해 판별력 지표로 AUC 사용)")

importances = pd.Series(rf.feature_importances_, index=feat_cols).sort_values(ascending=False)
print(importances.head(5))

plt.figure(figsize=(8, 5))
importances.head(10).iloc[::-1].plot(kind="barh", color="#3b6ea5")
plt.title(f"RandomForest 변수 중요도 Top 10 (DefectStatus 분류, AUC={auc:.3f})")
plt.xlabel("중요도")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/03_feature_importance.png", dpi=150)
plt.close()

# ------------------------------------------------------------------
# 4. Analyze — 1위 변수(MaintenanceHours) 구간별 불량률 패턴
# ------------------------------------------------------------------
top_var = importances.index[0]
bins = [0, 4, 8, 12, 16, 20, 24]
df["mh_bin"] = pd.cut(df[top_var], bins=bins)
bin_stats = df.groupby("mh_bin", observed=True)["DefectStatus"].agg(["mean", "count"])
print(f"\n[4] '{top_var}' 구간별 불량률:")
print(bin_stats)

THRESHOLD = 12  # bin_stats에서 불량률이 급격히 뛰는 경계값 (8~12군 79.7% -> 12~16군 95.3%)
high = df[df[top_var] > THRESHOLD].copy()
low = df[df[top_var] <= THRESHOLD].copy()
high_rate_actual = high.DefectStatus.mean()
low_rate_actual = low.DefectStatus.mean()
high_share = len(high) / n_total
print(f"[4] {top_var} > {THRESHOLD} 그룹: n={len(high)} ({high_share:.1%}), 실측 불량률={high_rate_actual:.3f}")
print(f"[4] {top_var} <= {THRESHOLD} 그룹: n={len(low)} ({1-high_share:.1%}), 실측 불량률={low_rate_actual:.3f}")

plt.figure(figsize=(7.5, 4.5))
bin_labels = [str(b) for b in bin_stats.index]
bars = plt.bar(bin_labels, bin_stats["mean"] * 100, color="#c0752f")
plt.axvline(2.5, color="#c0392b", ls="--", lw=1.3)
plt.text(2.55, 20, f"개선 목표 경계선\n({top_var} ≤ {THRESHOLD})", fontsize=9, color="#c0392b")
for b, v in zip(bars, bin_stats["mean"] * 100):
    plt.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}%", ha="center", fontsize=9)
plt.title(f"{top_var} 구간별 불량률 — 임계 구간 이후 급증 패턴")
plt.ylabel("불량률 (%)")
plt.xlabel(f"{top_var} 구간 (시간)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/04_defect_rate_vs_top_driver.png", dpi=150)
plt.close()

# ------------------------------------------------------------------
# 5. Improve — 모델 기반 반사실적(counterfactual) 개선 시뮬레이션
#    (실제 공정 개입이 아니라 학습된 모델로 "만약 조정했다면"을 추정)
# ------------------------------------------------------------------
target_value = low[top_var].median()
high_cf = high.copy()
high_cf[top_var] = target_value

proba_before = rf.predict_proba(high[feat_cols])[:, 1]
proba_after = rf.predict_proba(high_cf[feat_cols])[:, 1]
sim_before = proba_before.mean()
sim_after = proba_after.mean()
sim_pp_reduction = (sim_before - sim_after) * 100
sim_rel_reduction = (sim_before - sim_after) / sim_before * 100

overall_after = (low.DefectStatus.mean() * len(low) + proba_after.sum()) / n_total
overall_pp_reduction = (baseline_rate - overall_after) * 100
overall_rel_reduction = (baseline_rate - overall_after) / baseline_rate * 100

print(f"\n[5] 개선 시뮬레이션 — {top_var}를 {THRESHOLD}시간 초과 그룹({len(high)}건)에서 "
      f"목표값 {target_value:.0f}시간으로 조정 가정:")
print(f"    해당 그룹 모델추정 불량률: {sim_before:.3f} -> {sim_after:.3f} "
      f"({sim_pp_reduction:.1f}%p, 상대 {sim_rel_reduction:.1f}% 감소)")
print(f"    전체 공정 기준 모델추정 불량률: {baseline_rate:.3f} -> {overall_after:.3f} "
      f"({overall_pp_reduction:.1f}%p, 상대 {overall_rel_reduction:.1f}% 감소)")

plt.figure(figsize=(6.5, 5))
cats = ["개선 전\n(실측, 전체)", "개선 시뮬레이션 후\n(모델 추정, 전체)",
        f"개선 전\n({top_var}>{THRESHOLD} 그룹)", f"개선 후\n(모델 추정)"]
vals = [baseline_rate * 100, overall_after * 100, sim_before * 100, sim_after * 100]
colors = ["#8a8f98", "#2e7d32", "#c0392b", "#2e7d32"]
bars = plt.bar(cats, vals, color=colors)
for b, v in zip(bars, vals):
    plt.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold")
plt.ylabel("불량률 (%)")
plt.title("개선 조치 시뮬레이션 전/후 비교 (모델 기반 추정, 실제 개입 아님)")
plt.xticks(fontsize=8.5)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/05_before_after_simulation.png", dpi=150)
plt.close()

# ------------------------------------------------------------------
# 6. Control — 관리 규칙 제안 (요약 출력만, 별도 산출물 없음)
# ------------------------------------------------------------------
control_rule = (
    f"{top_var} 로트 평균이 {THRESHOLD}시간을 초과하면 즉시 경보. "
    f"p-관리도 UCL({ucl:.3f})/LCL({lcl:.3f}) 이탈 로트는 별도 원인조사."
)
print(f"\n[6] Control 제안: {control_rule}")

# ------------------------------------------------------------------
# 7. 요약 저장
# ------------------------------------------------------------------
summary = {
    "n_total": int(n_total),
    "n_features": int(len(feat_cols)),
    "baseline_defect_rate": float(baseline_rate),
    "spc": {
        "lot_size": LOT_SIZE,
        "n_lots": int(len(lot_stats)),
        "pbar": float(pbar),
        "ucl": float(ucl),
        "lcl": float(lcl),
        "n_out_of_control_lots": int(len(oc_lots)),
        "out_of_control_lot_ids": oc_lots.index.tolist(),
    },
    "analyze": {
        "defect_rate_linear_corr_note": "DefectRate(연속형)는 어떤 변수와도 |r|<0.03으로 사실상 무상관 (노이즈에 가까운 합성 변수)",
        "rf_auc_defect_status": float(auc),
        "top10_feature_importance": importances.head(10).to_dict(),
        "top_driver": top_var,
        "top_driver_bin_defect_rate": {str(k): float(v) for k, v in bin_stats["mean"].items()},
        "threshold": THRESHOLD,
        "high_group_n": int(len(high)),
        "high_group_share_pct": float(high_share * 100),
        "high_group_actual_defect_rate": float(high_rate_actual),
        "low_group_actual_defect_rate": float(low_rate_actual),
    },
    "improve_simulation": {
        "target_value": float(target_value),
        "high_group_before": float(sim_before),
        "high_group_after": float(sim_after),
        "high_group_pp_reduction": float(sim_pp_reduction),
        "high_group_rel_reduction_pct": float(sim_rel_reduction),
        "overall_before": float(baseline_rate),
        "overall_after": float(overall_after),
        "overall_pp_reduction": float(overall_pp_reduction),
        "overall_rel_reduction_pct": float(overall_rel_reduction),
        "disclosure": "실제 공정 개입 결과가 아니라 학습된 RandomForest 모델의 반사실적(counterfactual) 추정치임",
    },
    "control_rule": control_rule,
    "resume_highlights": {
        "dataset": "Kaggle 'Predicting Manufacturing Defects Dataset' 3,240건 17개 변수 (범용 제조공정 시뮬레이션 데이터)",
        "method": "p-관리도(SPC)로 관리이탈 로트 탐지 -> RandomForest 변수중요도로 불량 핵심 요인 규명 -> 반사실적 시뮬레이션으로 개선효과 정량화",
        "baseline_defect_rate_pct": round(baseline_rate * 100, 1),
        "top_driver_ko": "설비 유지보수시간(MaintenanceHours)" if top_var == "MaintenanceHours" else top_var,
        "top_driver": top_var,
        "threshold_desc": f"{top_var} {THRESHOLD}시간 초과 그룹(전체의 {high_share*100:.1f}%)의 실측 불량률 {high_rate_actual*100:.1f}% vs 이하 그룹 {low_rate_actual*100:.1f}%",
        "group_before_pct": round(sim_before * 100, 1),
        "group_after_pct": round(sim_after * 100, 1),
        "group_pp_reduction": round(sim_pp_reduction, 1),
        "group_rel_reduction_pct": round(sim_rel_reduction, 1),
        "overall_before_pct": round(baseline_rate * 100, 1),
        "overall_after_pct": round(overall_after * 100, 1),
        "overall_pp_reduction": round(overall_pp_reduction, 1),
        "overall_rel_reduction_pct": round(overall_rel_reduction, 1),
        "auc": round(auc, 3),
    },
}
with open("summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("\n분석 완료. figures/ 폴더와 summary.json을 확인하세요.")
