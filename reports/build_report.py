"""Builds reports/KNN_Project_Report.pdf from results_verified.json and reports/figures.

Run reports/analysis.py first, then:
    python reports/build_report.py
"""
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether, PageBreak,
                                PageTemplate, Paragraph, Spacer, Table, TableStyle)

OUT = Path(__file__).resolve().parent
R = json.loads((OUT / "results_verified.json").read_text(encoding="utf-8"))
FIG = OUT / "figures"
REPO = "https://github.com/Swayam424/breast-cancer-knn-optimization"

NAVY, INK, MUTED, RULE, TINT = map(colors.HexColor, ("#17324d", "#1f2933", "#5f6b76", "#d5dae0", "#f2f5f8"))
W = A4[0] - 44 * mm

body = ParagraphStyle("body", fontName="Helvetica", fontSize=9.7, leading=14.2, textColor=INK, spaceAfter=6, alignment=TA_LEFT)
small = ParagraphStyle("small", parent=body, fontSize=8.2, leading=11, textColor=MUTED, spaceAfter=3)
cell = ParagraphStyle("cell", parent=body, fontSize=8.3, leading=10.6, spaceAfter=0)
head = ParagraphStyle("head", parent=cell, fontName="Helvetica-Bold", textColor=NAVY)
h1 = ParagraphStyle("h1", parent=body, fontName="Helvetica-Bold", fontSize=13.5, leading=17, textColor=NAVY, spaceBefore=12, spaceAfter=6, keepWithNext=1)
h2 = ParagraphStyle("h2", parent=body, fontName="Helvetica-Bold", fontSize=10.3, leading=13.5, textColor=NAVY, spaceBefore=6, spaceAfter=3, keepWithNext=1)
caption = ParagraphStyle("caption", parent=small, spaceBefore=3, spaceAfter=9)
bullet = ParagraphStyle("bullet", parent=body, leftIndent=11, bulletIndent=1, spaceAfter=3)
title = ParagraphStyle("title", parent=body, fontName="Helvetica-Bold", fontSize=23, leading=28, textColor=NAVY, spaceAfter=6)
sub = ParagraphStyle("sub", parent=body, fontSize=11.5, leading=16, textColor=MUTED, spaceAfter=12)
stat_n = ParagraphStyle("statn", parent=body, fontName="Helvetica-Bold", fontSize=16, leading=19, textColor=NAVY, spaceAfter=1)
stat_l = ParagraphStyle("statl", parent=small, spaceAfter=0, fontSize=7.9, leading=10)


def P(t, s=body): return Paragraph(t, s)
def B(t): return Paragraph(t, bullet, bulletText="•")
def pct(v, d=2): return f"{v:.{d}f}%"


def table(rows, widths, align_from=1):
    data = [[P(str(c), head if i == 0 else cell) for c in row] for i, row in enumerate(rows)]
    t = Table(data, colWidths=[w * W for w in widths], repeatRows=1)
    t.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, 0), .9, NAVY), ("LINEBELOW", (0, 1), (-1, -1), .35, RULE),
                           ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 3.6), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.6),
                           ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 5)]))
    return t


def figure(name, cap, width=1.0):
    w, h = ImageReader(str(FIG / name)).getSize()
    return KeepTogether([Image(str(FIG / name), width=W * width, height=W * width * h / w), P(cap, caption)])


def row(name, m):
    return [name, pct(m["accuracy"]), pct(m["recall_malignant"]), f'{m["macro_f1"]:.4f}', m["fn"], m["fp"]]


E, T, CV, SK, SW, J, V, C = (R["experiments"], R["threshold"], R["repeated_cv"], R["soft_knn"], R["k_sweep"],
                             R["jitter_augmentation"], R["vectorisation"], R["soft_knn_curves"])
ADAM, RELU, SGDM, SGD = (SK[k] for k in ("AdamW, GELU, 60 epochs", "AdamW, ReLU, 60 epochs", "SGD + momentum, GELU, 60 epochs", "SGD, GELU, 60 epochs"))
EARLY = next(v for k, v in SK.items() if "early stop" in k)
ENS = R["soft_knn_ensemble_42_7_123"]
k7cv, t4, t3, t2 = CV["7"], T["4/7"], T["3/7"], T["2/7"]
gap1 = SW["train_acc"][0] - SW["cv_acc"][0]
gap7 = SW["train_acc"][6] - SW["cv_acc"][6]
gap25 = SW["train_acc"][24] - SW["cv_acc"][24]
sm = lambda s: f'{s["acc_mean"]:.2f}% ± {s["acc_std"]:.2f}'

story = []
stats = [(f'{k7cv["mean"]:.2f}%', f'K-NN (k=7) accuracy, repeated 5-fold CV (± {k7cv["std"]:.2f})'),
         (pct(E["Euclidean, k=7"]["accuracy"]), "K-NN (k=7) accuracy on the held-out test split"),
         (pct(t3["recall_malignant"]), f'Malignant recall with the 3-of-7 rule ({t3["fn"]} of {R["dataset"]["test_malignant"]} missed)'),
         (f'{gap1:.1f} pts', "Train-validation gap at k=1, the clearest overfitting signal")]
cards = Table([[[P(a, stat_n), P(b, stat_l)] for a, b in stats]], colWidths=[W / 4] * 4)
cards.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), TINT), ("LINEAFTER", (0, 0), (-2, -1), 3, colors.white), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 9), ("LEFTPADDING", (0, 0), (-1, -1), 8)]))

story += [Spacer(1, 8 * mm), P("ML THEORY ASSIGNMENT &nbsp;·&nbsp; PROJECT REPORT", ParagraphStyle("eyebrow", parent=small, fontName="Helvetica-Bold", textColor=MUTED, spaceAfter=9)),
          P("K-Nearest Neighbours for Breast Cancer Diagnosis", title),
          P("Implementation, optimisation and overfitting analysis on the Wisconsin Diagnostic Breast Cancer dataset", sub),
          P("<b>Swayam Mallick, Krishnendu Dhar, Rwitama Aon, Pragati Kumari</b><br/>Techno India University, West Bengal &nbsp;·&nbsp; Batch BCS 3B &nbsp;·&nbsp; November 2026"),
          P(f'Code: <link href="{REPO}" color="#2a78d6">github.com/Swayam424/breast-cancer-knn-optimization</link>', body), Spacer(1, 3 * mm), cards, Spacer(1, 2 * mm)]

story += [P("1. Summary", h1),
          P(f'This project applies the K-Nearest Neighbours (K-NN) classifier to a real clinical task: deciding whether a breast tumour is benign or malignant from measurements of its cell nuclei. Scope follows the updated assignment requirement, which specifies K-NN. '
            f'A standard K-NN with k=7 reaches {pct(E["Euclidean, k=7"]["accuracy"])} on the test split, and repeated cross-validation puts its expected accuracy at {k7cv["mean"]:.2f}% ± {k7cv["std"]:.2f}, the more reliable figure.'),
          P(f'The optimisation work shows that this dataset leaves little accuracy to gain: most tuned variants fall within the noise of the baseline. The useful gain is clinical. Lowering the malignant vote threshold from 4 of 7 to 3 of 7 catches one more cancer, at the cost of {t3["fp"] - t4["fp"]} false alarms. '
            f'A differentiable Soft-KNN raises recall further but overfits quickly: its training loss reaches almost zero while validation loss climbs.'),
          P("2. Real-life problem and purpose", h1),
          P("Fine-needle aspiration is a common way to examine a suspicious breast lump. The two possible errors are not equally serious: a false alarm leads to further tests, while a missed malignancy delays treatment. The project therefore tracks <b>malignant recall</b> alongside accuracy, and asks three questions:"),
          B("How accurately can K-NN separate malignant from benign tumours?"), B("Which optimisation techniques actually help, once random variation is accounted for?"), B("Where does the model overfit or underfit, and which settings avoid both?"),
          P("This is a coursework study on a small benchmark. It is not a validated medical device and must not guide patient care.", small),
          P("3. Dataset", h1),
          P(f'The Wisconsin Diagnostic Breast Cancer (WDBC) dataset holds {R["dataset"]["samples"]} tumour samples, {R["dataset"]["benign"]} benign and {R["dataset"]["malignant"]} malignant, each described by {R["dataset"]["features"]} continuous features such as radius, texture, concavity and smoothness. '
            'The dataset originates from the UCI Machine Learning Repository and is also published on Kaggle as <i>uciml/breast-cancer-wisconsin-data</i>. The code loads the identical data through scikit-learn, so no download is needed.'),
          P(f'Data is split 80/20 with stratification (seed 42), giving {R["dataset"]["train"]} training and {R["dataset"]["test"]} test samples, {R["dataset"]["test_malignant"]} of them malignant. Features range across several orders of magnitude, so they are standardised using statistics from the training data only.'),
          P("4. Method", h1),
          P("For a query sample, K-NN finds the <i>k</i> closest training samples and predicts the majority class among them. It has no training phase, so its behaviour is set by the choice of <i>k</i>, the distance metric, the feature space and the decision threshold. The pipeline runs in six steps:"),
          ]
for s in ["<b>Load and split</b> the data, with the test set held out.", "<b>Standardise</b> using training statistics only, refitting inside every cross-validation fold.",
          "<b>Benchmark distance metrics:</b> Euclidean, Manhattan, cosine and Mahalanobis.", "<b>Tune:</b> sweep <i>k</i> from 1 to 25 and compress features with PCA.",
          "<b>Optimise:</b> train a differentiable Soft-KNN, and compare optimisers, activations and augmentation.", "<b>Calibrate and evaluate:</b> set the vote threshold, then report accuracy, recall, macro F1 and error counts."]:
    story.append(B(s))
story += [P("<b>Soft-KNN.</b> Hard voting has no useful gradient, so it cannot be trained by backpropagation. Soft-KNN passes inputs through a small neural network (64 hidden units, 32-dimensional normalised output), then applies a softmax with temperature 0.1 to the negative mean distance to the 5 nearest samples of each class. The network can then learn a distance space using cross-entropy loss."),
          P("5. Baseline results", h1),
          table([["Configuration", "Test accuracy", "Malignant recall", "Macro F1", "Missed cancers", "False alarms"]] + [row(k, v) for k, v in E.items()], [.3, .15, .16, .12, .14, .13]),
          P(f'Table 1. K-NN on the held-out test split ({R["dataset"]["test"]} samples). PCA retains {R["pca_variance"]["2"]:.1f}%, {R["pca_variance"]["6"]:.1f}%, {R["pca_variance"]["10"]:.1f}% and {R["pca_variance"]["15"]:.1f}% of variance at d = 2, 6, 10 and 15.', caption),
          P(f'Euclidean, Manhattan and cosine distances perform alike. Mahalanobis is the outlier, missing {E["Mahalanobis, k=5"]["fn"]} of {R["dataset"]["test_malignant"]} cancers; a likely cause is that it fully whitens highly correlated features, which inflates low-variance noise directions. With a test split this small, one sample moves accuracy by 0.88 points, and the 95% margin around 97% is about ±3.1 points. Differences of a point or two in Table 1 should not be read as real.'),
          P("6. Optimisation", h1),
          P("The assignment asks how the model can be optimised through optimisers, activation functions, data augmentation and fine-tuning. Each neural result below is averaged over 10 random seeds, because a single seed can be misleading: Soft-KNN with AdamW spans "
            f'{ADAM["acc_min"]:.2f}% to {ADAM["acc_max"]:.2f}% depending on the seed.'),
          KeepTogether([table([["Technique", "Setting", "Test accuracy", "Malignant recall"],
                 ["Optimiser", "AdamW", sm(ADAM), f'{ADAM["recall_mean"]:.2f}%'], ["", "SGD + momentum", sm(SGDM), f'{SGDM["recall_mean"]:.2f}%'], ["", "SGD, no momentum", sm(SGD), f'{SGD["recall_mean"]:.2f}%'],
                 ["Activation", "ReLU (AdamW)", sm(RELU), f'{RELU["recall_mean"]:.2f}%'], ["", "GELU (AdamW)", sm(ADAM), f'{ADAM["recall_mean"]:.2f}%'],
                 ["Regularisation", f'Early stopping at epoch {C["best_val_loss_epoch"]}', sm(EARLY), f'{EARLY["recall_mean"]:.2f}%'],
                 ["Ensembling", "3 seeds, averaged", pct(ENS["accuracy"]), pct(ENS["recall_malignant"])],
                 ["Augmentation", "Gaussian jitter, std 0.1 (K-NN, k=7)", f'{J["0.1"]["acc_mean"]:.2f}% ± {J["0.1"]["acc_std"]:.2f}', f'{J["0.1"]["recall_mean"]:.2f}%'],
                 ["", "Gaussian jitter, std 0.3", f'{J["0.3"]["acc_mean"]:.2f}% ± {J["0.3"]["acc_std"]:.2f}', f'{J["0.3"]["recall_mean"]:.2f}%'],
                 ["", "Gaussian jitter, std 0.5", f'{J["0.5"]["acc_mean"]:.2f}% ± {J["0.5"]["acc_std"]:.2f}', f'{J["0.5"]["recall_mean"]:.2f}%']], [.2, .38, .22, .2]),
          P("Table 2. Optimisation experiments. Values show mean ± standard deviation over 10 seeds; the ensemble is one fixed combination of seeds 42, 7 and 123.", caption)]),
          B(f'<b>Optimisers.</b> AdamW and momentum outperform plain SGD, which also varies most between seeds. Plain SGD has not converged in 60 epochs at this learning rate.'),
          B("<b>Activation.</b> ReLU and GELU are indistinguishable given the seed-to-seed spread."),
          B("<b>Augmentation.</b> Jitter neither clearly helps nor seriously harms K-NN. Adding copies of the training data does not add new information about the tumours."),
          B(f'<b>Fine-tuning.</b> Cross-validation accuracy is nearly flat for k between 3 and 13, so any value there is defensible. Choosing k=7 is reasonable; the single best fold score at k={SW["best_cv_k"]} is within the noise.'),
          B(f'<b>Code speed.</b> Replacing Python loops with NumPy broadcasting returned identical predictions and ran about {V["speedup"]:.0f} times faster ({V["loop_ms"]:.0f} ms to {V["vector_ms"]:.0f} ms, median of 5 runs). The ratio depends on the machine.'),
          P("7. Loss and accuracy graphs", h1),
          P("K-NN has no epochs, so its curves are plotted against <i>k</i>, using zero-one loss (1 − accuracy). Soft-KNN is trained iteratively, so its curves are plotted against epochs, on a validation split carved from the training data so that the test set stays untouched."),
          figure("k_sweep_loss_accuracy.png", "Figure 1. K-NN training and 5-fold validation performance across k. The scaler is refit within each fold."),
          figure("softknn_loss_accuracy.png", "Figure 2. Soft-KNN with AdamW, averaged over 5 seeds, for 200 epochs."),
          P("8. Overfitting and underfitting", h1),
          P(f'<b>Overfitting at k=1.</b> Each training sample is its own nearest neighbour, so training accuracy is a perfect {SW["train_acc"][0]:.0f}%, while validation accuracy is {SW["cv_acc"][0]:.2f}%. This {gap1:.2f}-point gap is the signature of high variance: the model memorises noise.'),
          P(f'<b>Balanced at k=7.</b> Training ({SW["train_acc"][6]:.2f}%) and validation ({SW["cv_acc"][6]:.2f}%) sit close together with a {gap7:.2f}-point gap, while validation accuracy is near its plateau.'),
          P(f'<b>Mild underfitting at k=25.</b> Training accuracy falls to {SW["train_acc"][24]:.2f}% and validation to {SW["cv_acc"][24]:.2f}%. The gap is small ({gap25:.2f} points) but both scores are lower, which points to rising bias. The effect is mild because the two classes are well separated. Aggressive PCA gives a similar picture: at d=2, test accuracy drops to {pct(E["PCA d=2, k=7"]["accuracy"])}.'),
          P(f'<b>Overfitting in Soft-KNN.</b> Validation loss reaches its minimum of {C["best_val_loss"]:.3f} at epoch {C["best_val_loss_epoch"]} and then rises to {C["final_val_loss"]:.3f} by epoch 200, while training loss falls to almost zero and training accuracy reaches {C["final_train_acc"]:.0f}%. '
            f'Validation accuracy holds at {C["final_val_acc"]:.2f}%: the network keeps classifying well but becomes overconfident, which loss detects and accuracy does not. A model with thousands of parameters is memorising {R["dataset"]["train"]} samples. Stopping early at epoch {C["best_val_loss_epoch"]} did not raise test accuracy ({sm(EARLY)}) and gave lower recall, so its benefit here is lower validation loss, not more correct answers.'),
          P("9. Clinical threshold calibration", h1),
          P("Lowering the number of malignant votes needed to raise an alert trades false alarms for fewer missed cancers."),
          table([["Rule", "Test accuracy", "Malignant recall", "Macro F1", "Missed cancers", "False alarms"]] + [row(k + " votes", T[k]) for k in ("4/7", "3/7", "2/7")], [.3, .15, .16, .12, .14, .13]),
          P("Table 3. Effect of the malignant vote threshold, K-NN with k=7.", caption),
          figure("confusion_thresholds.png", "Figure 3. Confusion matrices for the default and safety rules.", .74),
          P(f'<b>Test patient #35</b> is a useful example. The tumour is malignant and receives {R["patient_35"]["malignant_votes_of_7"]} of 7 malignant votes, so the default rule misses it and the 3-of-7 rule catches it. The change does not remove all risk: two cancers with only one malignant vote each remain undetected, and lowering the threshold to 2 of 7 catches neither while adding more false alarms. '
            'Because this threshold was examined on the test split, the result is illustrative. A deployed threshold would need selecting on separate validation data, with clinicians weighing the cost of each error.'),
          P("10. Conclusions and recommendations", h1),
          B(f'<b>Recommended model:</b> K-NN with standardised features and k=7, with an expected accuracy of {k7cv["mean"]:.2f}% ± {k7cv["std"]:.2f}. It is simple, fast and explainable, and its overfitting risk is low.'),
          B(f'<b>Where missed cancers matter most:</b> consider the 3-of-7 rule, or the Soft-KNN ensemble, which had the fewest missed cancers ({ENS["fn"]} of {R["dataset"]["test_malignant"]}) with {ENS["fp"]} false alarms on this split.'),
          B("<b>Best next steps:</b> nested cross-validation for threshold selection, and external validation on data from another hospital, since one benchmark cannot show how the model generalises."),
          P("<b>Limitations.</b> The dataset is small and comes from a single source; test metrics carry wide margins, and the test split was consulted for several comparisons, so its figures may be slightly optimistic. The repeated cross-validation figures are the fairest estimate."),
          P("References", h1)]
for r in ["W. H. Wolberg, W. N. Street, O. L. Mangasarian. Breast Cancer Wisconsin (Diagnostic). UCI Machine Learning Repository, 1995.",
          "T. Cover, P. Hart. Nearest neighbor pattern classification. <i>IEEE Transactions on Information Theory</i>, 13(1), 1967.",
          "J. Goldberger, S. Roweis, G. Hinton, R. Salakhutdinov. Neighbourhood components analysis. <i>NeurIPS</i>, 2004.",
          "I. Loshchilov, F. Hutter. Decoupled weight decay regularization. <i>ICLR</i>, 2019.", "D. Hendrycks, K. Gimpel. Gaussian error linear units (GELUs). arXiv:1606.08415, 2016.",
          "F. Pedregosa et al. Scikit-learn: machine learning in Python. <i>JMLR</i>, 12, 2011."]:
    story.append(P(r, small))
story += [PageBreak(), P("Appendix A. Reproducibility and reconciliation", h1),
          P(f'Every figure in this report is generated by <font face="Courier">reports/analysis.py</font> and stored in <font face="Courier">reports/results_verified.json</font>, using Python {R["environment"]["python"]}, scikit-learn {R["environment"]["sklearn"]} and PyTorch {R["environment"]["torch"]} on CPU. Soft-KNN values may vary slightly across library versions, and timings across machines.'),
          P("To reproduce, run from the repository root:"),
          P('<font face="Courier" size="8.5">pip install -r requirements.txt<br/>python reports/analysis.py<br/>python reports/build_report.py</font>'),
          P("Recomputing the team's earlier paper confirmed the core K-NN, PCA, threshold and ensemble results. The items below differ, and this report's values replace them.", body),
          table([["Earlier claim", "Verified result", "Note"],
                 ["Convex fusion (E15): 97.37% accuracy", "Not verified", "No implementation is in the repository, and the results ledger records 94.74%. Excluded from this report."],
                 ["97.37% accuracy with 97.6% recall", "No single model", f'k=7 gives {pct(t4["accuracy"])} with {pct(t4["recall_malignant"])} recall; the ensemble gives {pct(ENS["accuracy"])} with {pct(ENS["recall_malignant"])} recall.'],
                 ["Zero-miss clinical recall", f'{t3["fn"]} cancers missed', f'The 3-of-7 rule reaches {pct(t3["recall_malignant"])} recall.'],
                 ["Soft-KNN AdamW: 96.49%", sm(ADAM), "96.49% is the best seed of 10, not the typical result."],
                 ["SGD without momentum: 91.23%", sm(SGD), "91.23% is the worst seed of 10."],
                 ["Vectorisation 122× faster", f'About {V["speedup"]:.0f}× here', "Timing depends on the machine. The 1974.8 ms figure listed for Euclidean distance matches the Python-loop benchmark, not the library K-NN used for the other metrics, so those latencies are not comparable."],
                 ["k=25 test accuracy: 94.74%", pct(SW["test_acc"][24]), "Recomputed with the same split and scaling."],
                 ["Gaussian jitter (std 0.3): 88.60%", f'{J["0.3"]["acc_mean"]:.2f}% ± {J["0.3"]["acc_std"]:.2f}', "Jitter added as extra training copies over 10 noise seeds; the earlier setup is not in the repository."]], [.33, .2, .47]),
          P("Table 4. Differences between the earlier paper and this report.", caption)]


def decorate(c, d):
    c.saveState(); c.setFont("Helvetica", 7.8); c.setFillColor(MUTED)
    c.drawString(22 * mm, 12 * mm, "K-Nearest Neighbours for Breast Cancer Diagnosis")
    c.drawRightString(A4[0] - 22 * mm, 12 * mm, f"Page {d.page}")
    c.restoreState()


doc = BaseDocTemplate(str(OUT / "KNN_Project_Report.pdf"), pagesize=A4, leftMargin=22 * mm, rightMargin=22 * mm, topMargin=19 * mm, bottomMargin=20 * mm,
                      title="K-Nearest Neighbours for Breast Cancer Diagnosis", author="Swayam Mallick, Krishnendu Dhar, Rwitama Aon, Pragati Kumari", subject="ML Theory Assignment project report")
doc.addPageTemplates([PageTemplate(id="p", frames=[Frame(doc.leftMargin, doc.bottomMargin, W, A4[1] - 39 * mm, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)], onPage=decorate)])
doc.build(story)
print("written", OUT / "KNN_Project_Report.pdf")
