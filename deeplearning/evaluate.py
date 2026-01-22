import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize

def evaluate(model, loader, device, class_names=None):
    model.eval()
    y_true, y_pred, y_probs = [], [], []

    with torch.no_grad():
        for num_x, cat_x, y in loader:
            num_x = num_x.to(device)
            cat_x = cat_x.to(device)

            outputs = model(num_x, cat_x)
            probs = torch.softmax(outputs, dim=1)

            y_true.extend(y.numpy())
            y_pred.extend(torch.argmax(probs, dim=1).cpu().numpy())
            y_probs.extend(probs.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)

    print("\nClassification Report")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            zero_division=0
        )
    )

    print("Confusion Matrix")
    print(confusion_matrix(y_true, y_pred))

    plot_roc(y_true, y_probs, class_names)

    
def plot_roc(y_true, y_probs, class_names=None):

    n_classes = len(class_names)

    plt.figure(figsize=(8, 6))

    # 🔵 BINARY CASE
    if n_classes == 2:
        # probability of positive class
        fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")

    # 🟢 MULTI-CLASS CASE
    else:
        y_true_bin = label_binarize(y_true, classes=range(n_classes))

        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
            roc_auc = auc(fpr, tpr)
            name = class_names[i] if class_names is not None else f"Class {i}"
            plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.show()
