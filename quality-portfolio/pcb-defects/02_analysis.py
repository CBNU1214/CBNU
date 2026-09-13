"""
PCB 결함 6종 자동 분류 (Missing_hole / Mouse_bite / Open_circuit / Short / Spur / Spurious_copper)
- 데이터: features_X.npy, labels_y.npy (01_prepare_data.py에서 생성, 2,953개 결함 인스턴스, 45차원)
- 특징: 기하/밝기 9 + HOG 36 (2x2 공간구역 x 9방향) = 45차원
- 목표: PCB 자동광학검사(AOI)에서 검출된 결함 후보를 유형별로 자동 분류(1차 트리아지)
"""
import json
import os
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image, ImageDraw
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

matplotlib.use("Agg")
warnings.filterwarnings("ignore")
from matplotlib import font_manager
_available_fonts = {f.name for f in font_manager.fontManager.ttflist}
_plot_font = next((f for f in ["AppleGothic", "Malgun Gothic", "Noto Sans CJK KR", "NanumGothic"] if f in _available_fonts), "DejaVu Sans")
sns.set_theme(style="whitegrid", font=_plot_font)
plt.rcParams["axes.unicode_minus"] = False

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

LABELS = ["missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper"]
LABELS_KO = {
    "missing_hole": "홀 누락", "mouse_bite": "마우스바이트", "open_circuit": "단선(Open)",
    "short": "단락(Short)", "spur": "돌기(Spur)", "spurious_copper": "동박 잔류",
}
CRITICAL = ["open_circuit", "short"]  # 전기적으로 즉시 기능 불량을 유발하는 치명적 결함

# ------------------------------------------------------------------
# 1. 데이터 로드
# ------------------------------------------------------------------
X = np.load("features_X.npy")
y = np.load("labels_y.npy")
meta = pd.read_pickle("data/instances_meta.pkl")
with open("data/feature_names.json", encoding="utf-8") as f:
    feature_names = json.load(f)

n_total = len(y)
class_counts = pd.Series(y).map(dict(enumerate(LABELS))).value_counts().reindex(LABELS)
print(f"[1] 결함 인스턴스 {n_total:,}개, 이미지 {meta.img_path.nunique()}장, 특징 {X.shape[1]}차원 로드")
print(class_counts)

# ------------------------------------------------------------------
# 2. 클래스 분포 시각화
# ------------------------------------------------------------------
plt.figure(figsize=(8, 4.5))
class_counts.plot(kind="bar", color="#3b6ea5")
plt.title(f"결함 유형별 인스턴스 수 (총 {n_total:,}개, 클래스당 {class_counts.min()}~{class_counts.max()}개로 균형)")
plt.ylabel("결함 인스턴스 수")
plt.xticks(rotation=20)
for i, v in enumerate(class_counts.values):
    plt.text(i, v + 3, str(v), ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/01_class_distribution.png", dpi=150)
plt.close()

# ------------------------------------------------------------------
# 3. 결함 유형별 대표 이미지 + bbox 시각화
# ------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.ravel()
for i, lbl in enumerate(LABELS):
    row = meta[meta.label == lbl].iloc[0]
    img = Image.open(row.img_path).convert("RGB")
    # 결함 주변을 크게 크롭한 뒤 bbox 표시 (원본은 3000px급이라 그대로 쓰면 결함이 점처럼 작음)
    pad = 150
    left = max(0, row.xmin - pad)
    upper = max(0, row.ymin - pad)
    right = min(img.width, row.xmax + pad)
    lower = min(img.height, row.ymax + pad)
    crop = img.crop((left, upper, right, lower))
    draw = ImageDraw.Draw(crop)
    draw.rectangle(
        [row.xmin - left, row.ymin - upper, row.xmax - left, row.ymax - upper],
        outline="red", width=4,
    )
    axes[i].imshow(crop)
    axes[i].set_title(f"{lbl} ({LABELS_KO[lbl]})", fontsize=13)
    axes[i].axis("off")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/02_sample_defects.png", dpi=150)
plt.close()
print("[2] 클래스 분포 및 대표 결함 이미지 시각화 완료")

# ------------------------------------------------------------------
# 4. 학습/평가 (Random Forest vs SVM)
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(f"[3] Train {len(X_train):,}개 / Test {len(X_test):,}개 (75/25 분할, stratify)")

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=400, random_state=42, n_jobs=-1, class_weight="balanced"
    ),
    "SVM (RBF, StandardScaler)": make_pipeline(
        StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale", random_state=42, class_weight="balanced")
    ),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    report = classification_report(
        y_test, y_pred, target_names=LABELS, output_dict=True, zero_division=0
    )
    results[name] = {"train_acc": train_acc, "test_acc": test_acc, "report": report, "y_pred": y_pred}
    print(f"[4] {name}: Train acc={train_acc:.4f}, Test acc={test_acc:.4f}, Macro F1={report['macro avg']['f1-score']:.4f}")

best_name = max(results, key=lambda n: results[n]["test_acc"])
best_pred = results[best_name]["y_pred"]
best_report = results[best_name]["report"]
print(f"[4] 최고 성능 모델: {best_name} (Test acc={results[best_name]['test_acc']:.4f})")

# ------------------------------------------------------------------
# 5. Confusion Matrix
# ------------------------------------------------------------------
cm = confusion_matrix(y_test, best_pred)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LABELS, yticklabels=LABELS, ax=axes[0])
axes[0].set_title(f"Confusion Matrix ({best_name})")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("True")
axes[0].tick_params(axis="x", rotation=30)
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=LABELS, yticklabels=LABELS, ax=axes[1])
axes[1].set_title("Normalized")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("True")
axes[1].tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/03_confusion_matrix.png", dpi=150)
plt.close()

# 가장 많이 혼동되는 클래스 쌍 찾기 (대각선 제외 최댓값)
cm_off = cm.copy().astype(float)
np.fill_diagonal(cm_off, 0)
i_true, i_pred = np.unravel_index(cm_off.argmax(), cm_off.shape)
top_confusion = {
    "true": LABELS[i_true], "pred": LABELS[i_pred], "count": int(cm[i_true, i_pred]),
}

# ------------------------------------------------------------------
# 6. 모델별 정확도 비교 + per-class F1/Recall
# ------------------------------------------------------------------
plt.figure(figsize=(6, 4.5))
names = list(results.keys())
test_accs = [results[n]["test_acc"] * 100 for n in names]
plt.bar(range(len(names)), test_accs, color=["#2e7d32", "#8a8f98"])
plt.xticks(range(len(names)), names, rotation=10)
plt.ylabel("Test Accuracy (%)")
plt.title("모델별 정확도 비교")
for i, v in enumerate(test_accs):
    plt.text(i, v + 0.5, f"{v:.1f}%", ha="center")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/04_model_comparison.png", dpi=150)
plt.close()

f1_per_class = {lbl: best_report[lbl]["f1-score"] for lbl in LABELS}
recall_per_class = {lbl: best_report[lbl]["recall"] for lbl in LABELS}
precision_per_class = {lbl: best_report[lbl]["precision"] for lbl in LABELS}

fig, ax1 = plt.subplots(figsize=(9, 4.5))
x = np.arange(len(LABELS))
width = 0.35
ax1.bar(x - width / 2, [f1_per_class[l] for l in LABELS], width, label="F1-score", color="#c0752f")
ax1.bar(x + width / 2, [recall_per_class[l] for l in LABELS], width, label="Recall", color="#3b6ea5")
ax1.set_xticks(x)
ax1.set_xticklabels(LABELS, rotation=20)
ax1.set_ylim(0, 1.05)
ax1.set_title(f"클래스별 F1-score / Recall ({best_name})")
ax1.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/05_f1_per_class.png", dpi=150)
plt.close()

hardest_class = min(f1_per_class, key=f1_per_class.get)
easiest_class = max(f1_per_class, key=f1_per_class.get)

# ------------------------------------------------------------------
# 7. Feature Importance (Random Forest 기준)
# ------------------------------------------------------------------
rf_model = models["Random Forest"]
importances = rf_model.feature_importances_
imp_series = pd.Series(importances, index=feature_names).sort_values(ascending=False)
top10 = imp_series.head(10)

plt.figure(figsize=(8, 5))
top10[::-1].plot(kind="barh", color="#2e7d32")
plt.title("Random Forest 특징 중요도 상위 10개")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/06_feature_importance.png", dpi=150)
plt.close()
print("[5] 시각화 완료 (confusion matrix, 모델비교, F1/Recall, feature importance)")

# ------------------------------------------------------------------
# 8. 요약 저장 (자소서/면접 재사용을 위한 핵심 숫자 포함)
# ------------------------------------------------------------------
critical_recall = {c: recall_per_class[c] for c in CRITICAL}

summary = {
    "n_images": int(meta.img_path.nunique()),
    "n_instances": int(n_total),
    "n_classes": len(LABELS),
    "class_distribution": class_counts.to_dict(),
    "n_features": int(X.shape[1]),
    "feature_breakdown": {"기하/밝기": 9, "HOG(2x2구역x9방향)": 36},
    "train_size": len(X_train),
    "test_size": len(X_test),
    "results": {
        name: {
            "train_acc": r["train_acc"],
            "test_acc": r["test_acc"],
            "macro_f1": r["report"]["macro avg"]["f1-score"],
            "weighted_f1": r["report"]["weighted avg"]["f1-score"],
            "macro_recall": r["report"]["macro avg"]["recall"],
        }
        for name, r in results.items()
    },
    "best_model": best_name,
    "per_class_f1_best_model": f1_per_class,
    "per_class_recall_best_model": recall_per_class,
    "per_class_precision_best_model": precision_per_class,
    "hardest_class": {"label": hardest_class, "f1": f1_per_class[hardest_class]},
    "easiest_class": {"label": easiest_class, "f1": f1_per_class[easiest_class]},
    "top_confusion_pair": top_confusion,
    "top10_feature_importance": top10.to_dict(),
    "critical_defect_recall": critical_recall,
    # --- 자소서/포트폴리오 재사용용 핵심 숫자 요약 ---
    "resume_highlights": {
        "dataset": "PCB 결함 이미지 693장 (PASCAL VOC 형식 bbox 주석), 6종 결함 총 2,953개 인스턴스",
        "feature_engineering": "결함 bbox를 64x64로 정규화 후 기하/밝기 특징 9개 + HOG(2x2구역x9방향) 36개, 총 45차원",
        "train_test_split": f"Train {len(X_train)}개 / Test {len(X_test)}개 (75/25, stratify)",
        "best_model": best_name,
        "best_test_accuracy_pct": round(results[best_name]["test_acc"] * 100, 1),
        "best_macro_f1": round(best_report["macro avg"]["f1-score"], 3),
        "hardest_class_ko": LABELS_KO[hardest_class],
        "hardest_class_f1": round(f1_per_class[hardest_class], 2),
        "easiest_class_ko": LABELS_KO[easiest_class],
        "easiest_class_f1": round(f1_per_class[easiest_class], 2),
        "top_confusion_desc": f"{LABELS_KO[top_confusion['true']]} -> {LABELS_KO[top_confusion['pred']]} 오분류 {top_confusion['count']}건 (최다)",
        "critical_defect_recall_desc": {LABELS_KO[c]: round(critical_recall[c] * 100, 1) for c in CRITICAL},
        "top1_feature": top10.index[0],
        "top1_feature_importance": round(float(top10.iloc[0]), 4),
    },
}
with open("summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("\n분석 완료. figures/ 폴더와 summary.json을 확인하세요.")
print(json.dumps(summary["resume_highlights"], ensure_ascii=False, indent=2))
