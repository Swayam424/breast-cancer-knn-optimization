"""Reproducible verification analysis for the WDBC K-NN study.

Recomputes the results used in the README and project report, and writes
reports/results_verified.json plus the charts in reports/figures/.

Run from the repository root:
    pip install -r requirements.txt
    python reports/analysis.py
"""
import json
import platform
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch
import torch.nn as nn
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_val_score, cross_validate, train_test_split)
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
torch.set_num_threads(4)

OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1f2933", "#5f6b76", "#e4e7eb"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#9aa5b1", "axes.linewidth": .8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": .7, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "axes.labelcolor": INK,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
    "xtick.color": MUTED, "ytick.color": MUTED, "legend.frameon": False, "savefig.dpi": 220,
    "savefig.bbox": "tight", "figure.facecolor": "white",
})

# Malignant is class 0 in scikit-learn and is the positive class for recall.
data = load_breast_cancer()
X, y = data.data, data.target
raw_tr, raw_te, ytr, yte = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
scaler = StandardScaler().fit(raw_tr)
Xtr, Xte = scaler.transform(raw_tr), scaler.transform(raw_te)
results = {"environment": {"python": platform.python_version(), "sklearn": sklearn.__version__, "torch": torch.__version__},
           "dataset": {"samples": len(y), "features": X.shape[1], "malignant": int((y == 0).sum()), "benign": int((y == 1).sum()),
                       "train": len(ytr), "test": len(yte), "test_malignant": int((yte == 0).sum())}}


def score(pred, mal_score=None):
    tn, fp, fn, tp = confusion_matrix((yte == 0).astype(int), (pred == 0).astype(int)).ravel()
    row = {"accuracy": accuracy_score(yte, pred) * 100, "recall_malignant": tp / (tp + fn) * 100,
           "precision_malignant": tp / (tp + fp) * 100 if tp + fp else 0.0, "macro_f1": f1_score(yte, pred, average="macro"),
           "fn": int(fn), "fp": int(fp), "tn": int(tn), "tp": int(tp)}
    if mal_score is not None:
        row["auroc"] = roc_auc_score((yte == 0).astype(int), mal_score)
    return row


def fit_score(model, tr=Xtr, te=Xte):
    model.fit(tr, ytr)
    return score(model.predict(te), model.predict_proba(te)[:, 0])


# Single-split experiments -------------------------------------------------
inv_cov = np.linalg.pinv(np.cov(Xtr, rowvar=False) + np.eye(30) * 1e-4)
exp = {"Euclidean, k=5": fit_score(KNN(5)), "Manhattan, k=5": fit_score(KNN(5, metric="manhattan")),
       "Cosine, k=5": fit_score(KNN(5, metric="cosine")),
       "Mahalanobis, k=5": fit_score(KNN(5, metric="mahalanobis", metric_params={"VI": inv_cov})),
       "Euclidean, k=7": fit_score(KNN(7))}
pca_var = {}
for d in (2, 6, 10, 15):
    p = PCA(d).fit(Xtr)
    pca_var[d] = p.explained_variance_ratio_.sum() * 100
    exp[f"PCA d={d}, k=7"] = fit_score(KNN(7), p.transform(Xtr), p.transform(Xte))
results["experiments"] = exp
results["pca_variance"] = pca_var

# Decision threshold -------------------------------------------------------
votes = KNN(7).fit(Xtr, ytr).predict_proba(Xte)[:, 0]
results["threshold"] = {f"{n}/7": score(np.where(votes >= n / 7 - 1e-9, 0, 1)) for n in (4, 3, 2)}
results["patient_35"] = {"true_class": "malignant" if yte[35] == 0 else "benign", "malignant_votes_of_7": int(round(votes[35] * 7))}
results["missed_at_3_of_7"] = [{"test_index": int(i), "malignant_votes_of_7": int(round(votes[i] * 7))}
                               for i in np.where((yte == 0) & (votes < 3 / 7 - 1e-9))[0]]

# Bias-variance sweep: the scaler is refit inside every fold ----------------
cv = StratifiedKFold(5, shuffle=True, random_state=42)
ks = list(range(1, 26))
train_acc, cv_acc, cv_std, test_acc = [], [], [], []
for k in ks:
    r = cross_validate(make_pipeline(StandardScaler(), KNN(k)), raw_tr, ytr, cv=cv, return_train_score=True)
    train_acc.append(r["train_score"].mean() * 100)
    cv_acc.append(r["test_score"].mean() * 100)
    cv_std.append(r["test_score"].std() * 100)
    test_acc.append(KNN(k).fit(Xtr, ytr).score(Xte, yte) * 100)
results["k_sweep"] = {"k": ks, "train_acc": train_acc, "cv_acc": cv_acc, "cv_std": cv_std, "test_acc": test_acc,
                      "best_cv_k": ks[int(np.argmax(cv_acc))]}
rep = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0)
results["repeated_cv"] = {}
for k in (1, 3, 5, 7, 9, 13, 25):
    s = cross_val_score(make_pipeline(StandardScaler(), KNN(k)), X, y, cv=rep) * 100
    results["repeated_cv"][k] = {"mean": s.mean(), "std": s.std()}

# Gaussian-jitter augmentation over 10 noise seeds --------------------------
jitter = {}
for std in (.1, .3, .5):
    accs, recs = [], []
    for sd in range(10):
        noise = np.random.RandomState(sd).normal(0, std, Xtr.shape)
        m = score(KNN(7).fit(np.vstack([Xtr, Xtr + noise]), np.r_[ytr, ytr]).predict(Xte))
        accs.append(m["accuracy"]); recs.append(m["recall_malignant"])
    jitter[std] = {"acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)), "recall_mean": float(np.mean(recs))}
results["jitter_augmentation"] = jitter

# Vectorisation timing, median of 5 runs ------------------------------------
def loop_predict():
    out = []
    for x in Xte:
        d = [np.sum((x - t) ** 2) ** .5 for t in Xtr]
        out.append(np.sum(ytr[np.argsort(d)[:7]]) > 3.5)
    return np.array(out, dtype=int)


def vector_predict():
    d = np.sqrt(np.sum((Xte[:, None, :] - Xtr[None, :, :]) ** 2, axis=2))
    return (np.sum(ytr[np.argsort(d, axis=1)[:, :7]], axis=1) > 3.5).astype(int)


def median_ms(fn):
    times = []
    for _ in range(5):
        t = time.perf_counter(); fn(); times.append((time.perf_counter() - t) * 1000)
    return float(np.median(times))


loop_ms, vec_ms = median_ms(loop_predict), median_ms(vector_predict)
results["vectorisation"] = {"loop_ms": loop_ms, "vector_ms": vec_ms, "speedup": loop_ms / vec_ms,
                            "same_predictions": bool((loop_predict() == vector_predict()).all())}


# Soft-KNN ------------------------------------------------------------------
class SoftKNN(nn.Module):
    def __init__(self, act):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(30, 64), nn.LayerNorm(64), act(), nn.Linear(64, 32))

    def forward(self, x):
        return nn.functional.normalize(self.net(x), p=2, dim=1)


def logits(dist, labels, tau=.1):
    return torch.stack([-dist[:, labels == c].topk(5, largest=False, dim=1).values.mean(1) / tau for c in (0, 1)], 1)


def make_optimizer(name, params):
    if name == "AdamW":
        return torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
    return torch.optim.SGD(params, lr=5e-3, momentum=.9 if name == "SGD + momentum" else 0.)


def train(seed, A, a, B, b, opt="AdamW", act=nn.GELU, epochs=60, track=False):
    """Train on (A, a); return malignant/benign posteriors for B and optional curves."""
    torch.manual_seed(seed)
    A, B = torch.tensor(A, dtype=torch.float32), torch.tensor(B, dtype=torch.float32)
    a, b = torch.tensor(a), torch.tensor(b)
    model, curves = SoftKNN(act), {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    optimizer = make_optimizer(opt, model.parameters())
    for _ in range(epochs):
        optimizer.zero_grad()
        z = model(A)
        lg = logits(torch.cdist(z, z) + torch.eye(len(z)) * 1e5, a)
        loss = nn.functional.cross_entropy(lg, a)
        loss.backward()
        optimizer.step()
        if track:
            with torch.no_grad():
                lv = logits(torch.cdist(model(B), model(A)), a)
                curves["train_loss"].append(loss.item())
                curves["val_loss"].append(nn.functional.cross_entropy(lv, b).item())
                curves["train_acc"].append((lg.argmax(1) == a).float().mean().item() * 100)
                curves["val_acc"].append((lv.argmax(1) == b).float().mean().item() * 100)
    with torch.no_grad():
        probs = torch.softmax(logits(torch.cdist(model(B), model(A)), a), 1).numpy()
    return probs, curves


def seeds_summary(**kw):
    rows = [score(train(s, Xtr, ytr, Xte, yte, **kw)[0].argmax(1)) for s in range(10)]
    acc, rec = [r["accuracy"] for r in rows], [r["recall_malignant"] for r in rows]
    return {"acc_mean": float(np.mean(acc)), "acc_std": float(np.std(acc)), "acc_min": min(acc), "acc_max": max(acc),
            "recall_mean": float(np.mean(rec)), "recall_std": float(np.std(rec))}


# Curves use a validation split carved from the training set: the test set is untouched.
fit_x, val_x, fit_y, val_y = train_test_split(raw_tr, ytr, test_size=.2, stratify=ytr, random_state=42)
s2 = StandardScaler().fit(fit_x)
fit_z, val_z = s2.transform(fit_x), s2.transform(val_x)
EPOCHS = 200
runs = [train(s, fit_z, fit_y, val_z, val_y, epochs=EPOCHS, track=True)[1] for s in range(5)]
mean = {k: np.mean([r[k] for r in runs], axis=0) for k in runs[0]}
best_epoch = int(np.argmin(mean["val_loss"])) + 1
results["soft_knn_curves"] = {"best_val_loss_epoch": best_epoch, "best_val_loss": float(mean["val_loss"][best_epoch - 1]),
                              "final_train_loss": float(mean["train_loss"][-1]), "final_val_loss": float(mean["val_loss"][-1]),
                              "final_train_acc": float(mean["train_acc"][-1]), "final_val_acc": float(mean["val_acc"][-1])}
results["soft_knn"] = {
    "AdamW, GELU, 60 epochs": seeds_summary(),
    "AdamW, ReLU, 60 epochs": seeds_summary(act=nn.ReLU),
    "SGD + momentum, GELU, 60 epochs": seeds_summary(opt="SGD + momentum"),
    "SGD, GELU, 60 epochs": seeds_summary(opt="SGD"),
    f"AdamW, GELU, early stop ({best_epoch} epochs)": seeds_summary(epochs=best_epoch),
}
ens = np.mean([train(s, Xtr, ytr, Xte, yte)[0] for s in (42, 7, 123)], 0)
results["soft_knn_ensemble_42_7_123"] = score(ens.argmax(1), ens[:, 0])


# Figures -------------------------------------------------------------------
def line_pair(ax, x, a, b, la, lb, ylabel, title):
    ax.plot(x, a, color=BLUE, lw=2, label=la)
    ax.plot(x, b, color=ORANGE, lw=2, ls=(0, (4, 2)), label=lb)
    ax.set_ylabel(ylabel); ax.set_title(title); ax.legend(loc="best")


ka = np.array(train_acc); ca = np.array(cv_acc)
fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
line_pair(ax[0], ks, ka, ca, "Training", "5-fold validation", "Accuracy (%)", "Accuracy vs. k")
line_pair(ax[1], ks, 1 - ka / 100, 1 - ca / 100, "Training", "5-fold validation", "Zero-one loss", "Loss vs. k")
for a_ in ax:
    a_.set_xlabel("Neighbours (k)"); a_.set_xticks([1, 5, 10, 15, 20, 25])
ax[0].annotate(f"k=1: {ka[0] - ca[0]:.1f}-point gap", xy=(1, ca[0]), xytext=(4.5, 94.9), color=INK, fontsize=8.5,
               arrowprops=dict(arrowstyle="-", color=MUTED, lw=.8))
fig.tight_layout(); fig.savefig(FIG / "k_sweep_loss_accuracy.png"); plt.close(fig)

ep = np.arange(1, EPOCHS + 1)
fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
line_pair(ax[0], ep, mean["train_loss"], mean["val_loss"], "Training", "Validation", "Cross-entropy loss", "Soft-KNN loss")
line_pair(ax[1], ep, mean["train_acc"], mean["val_acc"], "Training", "Validation", "Accuracy (%)", "Soft-KNN accuracy")
for a_ in ax:
    a_.set_xlabel("Epoch")
fig.tight_layout(); fig.savefig(FIG / "softknn_loss_accuracy.png"); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
for a_, key, title in zip(ax, ("4/7", "3/7"), ("Default rule: 4 of 7 votes", "Safety rule: 3 of 7 votes")):
    t = results["threshold"][key]
    M = np.array([[t["tn"], t["fp"]], [t["fn"], t["tp"]]])
    a_.imshow(M, cmap="Blues", vmin=0, vmax=90); a_.grid(False)
    for i in range(2):
        for j in range(2):
            a_.text(j, i, M[i, j], ha="center", va="center", fontsize=13, fontweight="bold", color="white" if M[i, j] > 45 else INK)
    a_.set_xticks([0, 1], ["Benign", "Malignant"]); a_.set_yticks([0, 1], ["Benign", "Malignant"])
    a_.set_xlabel("Predicted"); a_.set_ylabel("Actual"); a_.set_title(title)
    for s in a_.spines.values():
        s.set_visible(False)
fig.tight_layout(); fig.savefig(FIG / "confusion_thresholds.png"); plt.close(fig)

cum = np.cumsum(PCA().fit(Xtr).explained_variance_ratio_) * 100
fig, a_ = plt.subplots(figsize=(4.8, 2.8))
a_.plot(range(1, 31), cum, color=BLUE, lw=2)
a_.scatter([10], [cum[9]], s=36, color=BLUE, edgecolor="white", zorder=3)
a_.annotate(f"d=10: {cum[9]:.1f}%", xy=(10, cum[9]), xytext=(12, cum[9] - 13), color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=.8))
a_.set_xlabel("Principal components (d)"); a_.set_ylabel("Cumulative variance (%)"); a_.set_title("PCA variance retained")
fig.tight_layout(); fig.savefig(FIG / "pca_variance.png"); plt.close(fig)

(OUT / "results_verified.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print(json.dumps({k: v for k, v in results.items() if k not in ("k_sweep", "experiments", "threshold")}, indent=1))
