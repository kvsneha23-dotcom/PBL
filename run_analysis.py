"""
run_analysis.py
───────────────────────────────────────────────────────────────────────────────
Local analysis script — generates all charts as PNGs and a metrics summary CSV.
Run from the project root:
    python run_analysis.py
───────────────────────────────────────────────────────────────────────────────
"""

import warnings, os
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix, roc_curve, auc)
from sklearn.feature_selection import mutual_info_classif
from scipy import stats

# ── Output directory ────────────────────────────────────────────────────────
OUT = "analysis_output"
os.makedirs(OUT, exist_ok=True)
print(f"Output directory: {OUT}/")

# ── Load data ────────────────────────────────────────────────────────────────
DATA_PATH = "dashboard/data/india_fb_platform_cleaned.csv"
if not os.path.exists(DATA_PATH):
    DATA_PATH = "india_fb_platform_cleaned.csv"

df = pd.read_csv(DATA_PATH)
n = len(df)
print(f"Loaded: {df.shape[0]} rows × {df.shape[1]} cols")

# ── Feature engineering (mirrors app.py) ────────────────────────────────────
df["adoption_intent"] = (df["adopt_likelihood"] >= 4).astype(int)

device_map = {"Smartphone": 1.0, "Tablet": 0.75,
              "Laptop/PC": 0.75, "Basic Phone": 0.25}
df["device_score"] = df["primary_device"].map(device_map).fillna(0.5)

dc_n   = (df["digital_comfort"] - 1) / 4
sa_n   = df["smartphone_access"].fillna(df["smartphone_access"].median()) / 3
df["digital_readiness_index"] = (0.45*dc_n + 0.35*df["device_score"] + 0.20*sa_n).clip(0,1)

mask_emp = df["survey_type"] == "employer"
mask_wkr = df["survey_type"] == "worker"

dtf_n = ((df["days_to_fill"] - 2) / 35).clip(0,1)
dro_n = (df["dropout_rate"] / 4).clip(0,1)
hfq_n = (df["hire_freq"] / 4).clip(0,1)
df["hiring_difficulty_score"] = np.where(mask_emp,
    0.40*dtf_n + 0.35*dro_n + 0.25*hfq_n, 0.0)

gti_n = (df["govt_trust_impact"].fillna(3) - 1) / 4
gei_n = (df["govt_endorsement_imp"].fillna(3) - 1) / 4
rtr_n = (df["review_system_trust"].fillna(3) - 1) / 4
dpa_n = (df["dpdpa_importance"].fillna(3) - 1) / 4
df["trust_score"] = (0.30*gti_n + 0.30*gei_n + 0.25*rtr_n + 0.15*dpa_n).clip(0,1)

ps_n  = (df["pain_severity"] - 1) / 4
ttj_n = (df["time_to_find_job"].fillna(2) / 4).clip(0,1)
jcf_n = (df["job_change_freq"].fillna(2) / 4).clip(0,1)
df["labour_friction_score"] = np.where(
    mask_wkr, 0.40*ps_n + 0.35*ttj_n + 0.25*jcf_n,
    np.where(mask_emp, 0.40*ps_n + 0.35*dtf_n.fillna(0) + 0.25*dro_n.fillna(0),
             0.70*ps_n + 0.15)).clip(0,1)

tl_map = {"Immediate":1.0,"1-3 months":0.8,"3-6 months":0.6,
           "6-12 months":0.4,"12+ months":0.2}
tl_n = df["adoption_timeline_emp"].map(tl_map).fillna(0)
wtp_n = (df["max_wtp_employer_tier"].fillna(0) / 5).clip(0,1)
csat_inv = 1 - ((df["curr_satisfaction"] - 1) / 4)
df["employer_adoption_score"] = np.where(mask_emp,
    0.40*tl_n + 0.35*wtp_n + 0.25*csat_inv, 0.0)

edu_n  = (df["education"].fillna(3) / 6).clip(0,1)
sa2_n  = (df["smartphone_access"].fillna(1.5) / 3).clip(0,1)
df["worker_employability_score"] = np.where(mask_wkr,
    0.40*edu_n + 0.35*sa2_n + 0.25*dc_n, 0.0)

df["wtp_unified"] = np.where(mask_emp,
    df["max_wtp_employer_tier"].fillna(0) / 5,
    np.where(mask_wkr, df["max_fee_worker_tier"].fillna(0) / 4, 0.0))

pp_cols = [c for c in df.columns if c.startswith("pp_") or c.startswith("wp_")]
df["pain_point_count"] = df[pp_cols].fillna(0).sum(axis=1)

COMP_SCORES = ["digital_readiness_index","hiring_difficulty_score","trust_score",
               "labour_friction_score","employer_adoption_score",
               "worker_employability_score","wtp_unified","pain_point_count"]

SEGMENT_COLORS = {"InnovationChampion":"#2196F3","CostConscious":"#4CAF50",
                  "DigitalLaggard":"#FF9800","StatusQuo":"#9C27B0"}
SURVEY_COLORS  = {"employer":"#1565C0","worker":"#2E7D32",
                  "contractor":"#E65100","govt":"#6A1B9A"}

sns.set_theme(style="whitegrid", palette="Set2", font_scale=1.0)

# ============================================================================
# FIGURE 1 — Survey type & Segment distribution
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

vc = df["survey_type"].value_counts()
axes[0].bar(vc.index, vc.values,
            color=[SURVEY_COLORS.get(k, "#888") for k in vc.index])
axes[0].set_title("Respondents by Survey Type", fontweight="bold")
axes[0].set_ylabel("Count")
for i, (k, v) in enumerate(vc.items()):
    axes[0].text(i, v + 5, f"{v} ({v/n*100:.0f}%)", ha="center", fontsize=9)

seg_vc = df["segment"].value_counts()
colors_seg = [SEGMENT_COLORS.get(k, "#888") for k in seg_vc.index]
axes[1].barh(seg_vc.index, seg_vc.values, color=colors_seg)
axes[1].set_title("Respondents by Behavioural Segment", fontweight="bold")
axes[1].set_xlabel("Count")

plt.tight_layout()
plt.savefig(f"{OUT}/fig1_distribution.png", dpi=130, bbox_inches="tight")
plt.close()
print("✔  fig1_distribution.png")

# ============================================================================
# FIGURE 2 — Adoption Intent cross-tabs
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ct1 = pd.crosstab(df["segment"], df["adoption_intent"], normalize="index")*100
ct1.columns = ["Low Intent", "High Intent"]
ct1.plot(kind="bar", stacked=True, color=["#ef5350","#66bb6a"],
         ax=axes[0], legend=True, rot=25)
axes[0].set_title("Adoption Intent by Segment", fontweight="bold")
axes[0].set_ylabel("% Respondents")
axes[0].set_ylim(0, 115)

ct2 = pd.crosstab(df["survey_type"], df["adoption_intent"], normalize="index")*100
ct2.columns = ["Low Intent", "High Intent"]
ct2.plot(kind="bar", stacked=True, color=["#ef5350","#66bb6a"],
         ax=axes[1], legend=True, rot=0)
axes[1].set_title("Adoption Intent by Stakeholder Type", fontweight="bold")
axes[1].set_ylabel("% Respondents")

plt.tight_layout()
plt.savefig(f"{OUT}/fig2_adoption_crosstabs.png", dpi=130, bbox_inches="tight")
plt.close()
print("✔  fig2_adoption_crosstabs.png")

# ============================================================================
# FIGURE 3 — Key metrics distributions
# ============================================================================
cols_dist = ["digital_comfort","adopt_likelihood","pain_severity",
             "curr_satisfaction","nps_score"]
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, col in enumerate(cols_dist):
    axes[i].hist(df[col].dropna(), bins=10,
                 color=sns.color_palette("Set2")[i], edgecolor="white")
    axes[i].set_title(col.replace("_"," ").title(), fontweight="bold")
    axes[i].set_xlabel("Value")
    axes[i].set_ylabel("Count")
    mu = df[col].mean()
    axes[i].axvline(mu, color="red", linestyle="--", linewidth=1.2,
                    label=f"μ={mu:.2f}")
    axes[i].legend(fontsize=8)

axes[5].axis("off")
plt.suptitle("Key Metric Distributions", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{OUT}/fig3_distributions.png", dpi=130, bbox_inches="tight")
plt.close()
print("✔  fig3_distributions.png")

# ============================================================================
# FIGURE 4 — Composite score box plots by segment
# ============================================================================
display_scores = ["digital_readiness_index","trust_score",
                  "labour_friction_score","wtp_unified"]
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
axes = axes.flatten()
segs = sorted(df["segment"].unique())
seg_colors = [SEGMENT_COLORS.get(s, "#888") for s in segs]

for i, score in enumerate(display_scores):
    data = [df[df["segment"] == s][score].dropna().values for s in segs]
    bp = axes[i].boxplot(data, patch_artist=True, labels=segs,
                         medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], seg_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[i].set_title(score.replace("_"," ").title(), fontweight="bold")
    axes[i].set_ylabel("Score")
    axes[i].tick_params(axis="x", rotation=20)

plt.suptitle("Composite Score Distributions by Segment", fontsize=14,
             fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{OUT}/fig4_composite_scores_boxplot.png", dpi=130, bbox_inches="tight")
plt.close()
print("✔  fig4_composite_scores_boxplot.png")

# ============================================================================
# FIGURE 5 — Correlation heatmap
# ============================================================================
hm_cols = ["adopt_likelihood","digital_readiness_index","hiring_difficulty_score",
           "trust_score","labour_friction_score","employer_adoption_score",
           "worker_employability_score","wtp_unified","pain_severity","nps_score"]
corr = df[hm_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, linewidths=0.5, ax=ax, annot_kws={"size": 8},
            vmin=-1, vmax=1)
ax.set_title("Correlation Heatmap — Business Features & Platform Metrics",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT}/fig5_correlation_heatmap.png", dpi=130, bbox_inches="tight")
plt.close()
print("✔  fig5_correlation_heatmap.png")

# ============================================================================
# FIGURE 6 — Adoption rate by Digital Readiness level
# ============================================================================
dri_bins = pd.cut(df["digital_readiness_index"], 5,
                  labels=["Very Low","Low","Moderate","High","Very High"])
dri_adopt = df.groupby(dri_bins.astype(str))["adoption_intent"].mean() * 100

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(dri_adopt.index, dri_adopt.values,
              color=sns.color_palette("Blues", len(dri_adopt)))
for bar, val in zip(bars, dri_adopt.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=10)
ax.set_title("Platform Adoption Rate by Digital Readiness Level",
             fontweight="bold")
ax.set_xlabel("Digital Readiness Index Bucket")
ax.set_ylabel("Adoption Rate (%)")
ax.set_ylim(0, 100)
plt.tight_layout()
plt.savefig(f"{OUT}/fig6_dri_vs_adoption.png", dpi=130, bbox_inches="tight")
plt.close()
print("✔  fig6_dri_vs_adoption.png")

# ============================================================================
# FIGURE 7 — WTP analysis
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

wtp_e = df[mask_emp]["max_wtp_employer_tier"].value_counts().sort_index()
axes[0].bar(wtp_e.index.astype(str), wtp_e.values, color="#1565C0", alpha=0.8)
axes[0].set_title("Employer WTP Tier Distribution", fontweight="bold")
axes[0].set_xlabel("WTP Tier (0=None → 5=High)")
axes[0].set_ylabel("Count")

wtp_w = df[mask_wkr]["willing_to_pay_worker"].value_counts()
axes[1].pie(wtp_w.values, labels=wtp_w.index,
            colors=["#66bb6a","#ef5350","#ff9800"],
            autopct="%1.1f%%", startangle=90)
axes[1].set_title("Worker Willingness to Pay", fontweight="bold")

plt.tight_layout()
plt.savefig(f"{OUT}/fig7_wtp_analysis.png", dpi=130, bbox_inches="tight")
plt.close()
print("✔  fig7_wtp_analysis.png")

# ============================================================================
# FIGURE 8 — Pain points frequency
# ============================================================================
pp_e_cols = [c for c in df.columns if c.startswith("pp_")]
wp_w_cols = [c for c in df.columns if c.startswith("wp_")]

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

if pp_e_cols:
    pp_emp = df[mask_emp][pp_e_cols].fillna(0).mean().sort_values()
    labels = [c.replace("pp_","").replace("_"," ").title() for c in pp_emp.index]
    axes[0].barh(labels, pp_emp.values * 100, color="#1565C0", alpha=0.8)
    axes[0].set_title("Employer Pain Points (% Flagged)", fontweight="bold")
    axes[0].set_xlabel("% Employers Reporting")

if wp_w_cols:
    wp_wkr = df[mask_wkr][wp_w_cols].fillna(0).mean().sort_values()
    labels2 = [c.replace("wp_","").replace("_"," ").title() for c in wp_wkr.index]
    axes[1].barh(labels2, wp_wkr.values * 100, color="#2E7D32", alpha=0.8)
    axes[1].set_title("Worker Pain Points (% Flagged)", fontweight="bold")
    axes[1].set_xlabel("% Workers Reporting")

plt.tight_layout()
plt.savefig(f"{OUT}/fig8_pain_points.png", dpi=130, bbox_inches="tight")
plt.close()
print("✔  fig8_pain_points.png")

# ============================================================================
# ML PIPELINE
# ============================================================================
print("\n🤖 Running ML pipeline…")

# Build feature matrix (same as app.py)
st_dummies  = pd.get_dummies(df["survey_type"], prefix="stype", dtype=float)
seg_dummies = pd.get_dummies(df["segment"],      prefix="seg",   dtype=float)
le = LabelEncoder()
df["state_le"] = le.fit_transform(df["state"].fillna("Unknown"))

UNIVERSAL = ["digital_comfort","pain_severity","curr_satisfaction","nps_score"]
feature_df = pd.concat([
    df[COMP_SCORES + UNIVERSAL + ["state_le"]],
    st_dummies, seg_dummies
], axis=1).fillna(0)

y = df["adoption_intent"]
X = feature_df.values
scaler = StandardScaler()
X_sc = scaler.fit_transform(X)

X_tr, X_te, y_tr, y_te = train_test_split(
    X_sc, y, test_size=0.25, random_state=42, stratify=y)

models = {
    "KNN":            KNeighborsClassifier(n_neighbors=7),
    "Decision Tree":  DecisionTreeClassifier(max_depth=6, random_state=42),
    "Random Forest":  RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42),
    "Gradient Boost": GradientBoostingClassifier(n_estimators=150, max_depth=4,
                                                  learning_rate=0.1, random_state=42),
}

results = {}
for name, clf in models.items():
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    y_prob = clf.predict_proba(X_te)[:, 1]
    cm = confusion_matrix(y_te, y_pred)
    tn, fp, fn, tp = cm.ravel()
    n_te = len(y_te)
    fpr_r, tpr_r, _ = roc_curve(y_te, y_prob)
    roc_auc = auc(fpr_r, tpr_r)
    cv_f1 = cross_val_score(
        models[name].__class__(**models[name].get_params()),
        X_sc, y, cv=5, scoring="f1").mean()
    results[name] = {
        "accuracy": accuracy_score(y_te, y_pred),
        "precision": precision_score(y_te, y_pred, zero_division=0),
        "recall": recall_score(y_te, y_pred, zero_division=0),
        "f1": f1_score(y_te, y_pred, zero_division=0),
        "roc_auc": roc_auc,
        "cv_f1": cv_f1,
        "cm": cm,
        "fpr": fpr_r, "tpr": tpr_r,
        "fp_pct": fp/n_te*100, "fn_pct": fn/n_te*100,
        "fp_of_pos": fp/(fp+tp+1e-9)*100,
        "fn_of_neg": fn/(fn+tn+1e-9)*100,
    }
    if hasattr(clf, "feature_importances_"):
        results[name]["importances"] = clf.feature_importances_
    print(f"  {name:18s} Acc={results[name]['accuracy']:.4f}  "
          f"F1={results[name]['f1']:.4f}  "
          f"AUC={results[name]['roc_auc']:.4f}  "
          f"CV-F1={results[name]['cv_f1']:.4f}  "
          f"FP%={results[name]['fp_pct']:.1f}  "
          f"FN%={results[name]['fn_pct']:.1f}")

# ── Metrics CSV ──────────────────────────────────────────────────────────────
metrics_df = pd.DataFrame([{
    "Model": name,
    "Accuracy": f"{r['accuracy']:.4f}",
    "Precision": f"{r['precision']:.4f}",
    "Recall": f"{r['recall']:.4f}",
    "F1-Score": f"{r['f1']:.4f}",
    "ROC-AUC": f"{r['roc_auc']:.4f}",
    "CV F1 (5-fold)": f"{r['cv_f1']:.4f}",
    "FP % of test": f"{r['fp_pct']:.1f}%",
    "FN % of test": f"{r['fn_pct']:.1f}%",
} for name, r in results.items()])
metrics_df.to_csv(f"{OUT}/ml_metrics.csv", index=False)
print(f"\n✔  ml_metrics.csv saved")

# ============================================================================
# FIGURE 9 — Confusion matrices (2×2 grid)
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(11, 9))
axes = axes.flatten()
model_names = list(results.keys())

for i, name in enumerate(model_names):
    cm = results[name]["cm"]
    n_te = cm.sum()
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues",
                ax=axes[i], linewidths=0.5, linecolor="white")
    # Add custom annotations with %
    for row in range(2):
        for col in range(2):
            val = cm[row, col]
            axes[i].text(col+0.5, row+0.5,
                         f"{val}\n({val/n_te*100:.1f}%)",
                         ha="center", va="center",
                         fontsize=11, fontweight="bold",
                         color="white" if cm[row, col] > cm.max()/2 else "black")
    axes[i].set_title(f"{name}", fontweight="bold", fontsize=11)
    axes[i].set_xlabel("Predicted")
    axes[i].set_ylabel("Actual")
    axes[i].set_xticklabels(["Low Intent", "High Intent"], fontsize=9)
    axes[i].set_yticklabels(["Low Intent", "High Intent"], fontsize=9, rotation=0)

plt.suptitle("Confusion Matrices — All Models (with % of test set)",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{OUT}/fig9_confusion_matrices.png", dpi=130, bbox_inches="tight")
plt.close()
print("✔  fig9_confusion_matrices.png")

# ============================================================================
# FIGURE 10 — ROC curves
# ============================================================================
fig, ax = plt.subplots(figsize=(8, 7))
colors_roc = ["#2196F3","#4CAF50","#FF9800","#9C27B0"]
ax.plot([0,1],[0,1],"--",color="grey",label="Random (AUC=0.50)")

for i, name in enumerate(model_names):
    r = results[name]
    ax.plot(r["fpr"], r["tpr"], color=colors_roc[i], linewidth=2,
            label=f"{name} (AUC={r['roc_auc']:.3f})")

ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curves — All Classification Models", fontsize=13, fontweight="bold")
ax.legend(loc="lower right", fontsize=10)
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
plt.tight_layout()
plt.savefig(f"{OUT}/fig10_roc_curves.png", dpi=130, bbox_inches="tight")
plt.close()
print("✔  fig10_roc_curves.png")

# ============================================================================
# FIGURE 11 — Feature importance (RF)
# ============================================================================
if "importances" in results["Random Forest"]:
    imp_df = pd.DataFrame({
        "Feature": feature_df.columns,
        "Importance": results["Random Forest"]["importances"]
    }).sort_values("Importance", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(data=imp_df, x="Importance", y="Feature", palette="Blues_r", ax=ax)
    ax.set_title("Random Forest — Top 15 Feature Importances", fontweight="bold")
    ax.set_xlabel("Feature Importance")
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig11_feature_importance.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("✔  fig11_feature_importance.png")

# ============================================================================
# FIGURE 12 — Mutual Information
# ============================================================================
mi = mutual_info_classif(feature_df.fillna(0), y, random_state=42)
mi_df = pd.DataFrame({"Feature": feature_df.columns, "MI Score": mi})
mi_df = mi_df.sort_values("MI Score", ascending=False).head(15)

fig, ax = plt.subplots(figsize=(9, 6))
sns.barplot(data=mi_df, x="MI Score", y="Feature", palette="Greens_r", ax=ax)
ax.set_title("Top 15 Features — Mutual Information with Adoption Intent",
             fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT}/fig12_mutual_information.png", dpi=130, bbox_inches="tight")
plt.close()
print("✔  fig12_mutual_information.png")

# ============================================================================
# FIGURE 13 — Composite score profiles by segment (radar-like grouped bar)
# ============================================================================
disp_scores = ["digital_readiness_index","trust_score",
               "labour_friction_score","employer_adoption_score",
               "worker_employability_score","wtp_unified"]
seg_scores = df.groupby("segment")[disp_scores].mean()
seg_scores_n = (seg_scores - seg_scores.min()) / (seg_scores.max()-seg_scores.min()+1e-9)

fig, ax = plt.subplots(figsize=(13, 6))
x = np.arange(len(disp_scores))
width = 0.18
segs = seg_scores_n.index.tolist()
for k, seg in enumerate(segs):
    offset = (k - (len(segs)-1)/2) * width
    ax.bar(x + offset, seg_scores_n.loc[seg], width=width,
           label=seg, color=SEGMENT_COLORS.get(seg, "#888"), alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels([s.replace("_"," ").title() for s in disp_scores], rotation=25, ha="right")
ax.set_ylabel("Normalised Score [0-1]")
ax.set_title("Composite Score Profiles by Segment", fontweight="bold")
ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig(f"{OUT}/fig13_segment_profiles.png", dpi=130, bbox_inches="tight")
plt.close()
print("✔  fig13_segment_profiles.png")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
print(f"Dataset     : {n} respondents, {df.shape[1]} features")
print(f"Target dist : {y.sum()} High Intent ({y.mean()*100:.1f}%), "
      f"{(1-y).sum()} Low Intent ({(1-y.mean())*100:.1f}%)")
print("\nML Results:")
for name, r in results.items():
    print(f"  {name:18s}  Acc={r['accuracy']:.4f}  F1={r['f1']:.4f}  "
          f"AUC={r['roc_auc']:.4f}")
print(f"\nOutputs saved to: {OUT}/")
files = sorted(os.listdir(OUT))
for f in files:
    print(f"  {f}")
