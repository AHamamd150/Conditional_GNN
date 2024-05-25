# Generic imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Scikit-learn imports
from sklearn.metrics import roc_curve, auc

# PyTorch
import torch

# PyTorch Lightning imports
from lightning.pytorch.callbacks import Callback

# Torchmetrics
from torchmetrics.classification import BinaryConfusionMatrix, ConfusionMatrix


class TestCallback(Callback):
    def on_test_epoch_end(self, trainer, module):
        predictions = np.asarray(module.test_predictions)
        targets = np.asarray(module.test_targets)

        # --------------------------------------------------------------------------------
        # ROC Curve
        # --------------------------------------------------------------------------------
        # Softmax  predictions[:, 1]
        # create ROC curve
        fpr, tpr, threshold = roc_curve(targets, predictions)

        # save the numpy array
        # np.save("tpr_" + str(module.angle) + ".npy", tpr)
        # np.save("fpr_" + str(module.angle) + ".npy", fpr)
        # np.save("predictions_" + str(module.angle) + ".npy", predictions)
        # np.save("targets_" + str(module.angle) + ".npy", targets)

        plt.plot(
            fpr,
            tpr,
            linewidth=2,
            color="b",
            label="GNN (area = {:.3f}%)".format(auc(fpr, tpr) * 100),
        )
        plt.plot([0, 1], [0, 1], "k--", linewidth=1)
        plt.xlabel("False Positive")
        plt.ylabel("True Positive")
        plt.title("ROC curve")
        plt.grid(linestyle="--", color="k", linewidth=1.1)
        plt.legend(fontsize=20)
        plt.tight_layout()
        plt.savefig(trainer.log_dir + "_roc.png", dpi=300)

        # --------------------------------------------------------------------------------
        # Confusion Matrix
        # --------------------------------------------------------------------------------
        # Softmax
        # predictions = np.argmax(predictions, axis=1)
        confmat = BinaryConfusionMatrix(normalize="pred")
        # confmat = ConfusionMatrix(
        #    task="multiclass", num_classes=module.num_classes, normalize="pred"
        # )
        cm = confmat(torch.tensor(predictions), torch.tensor(targets)).numpy()

        figcm, ax = plt.subplots(figsize=(7, 5))
        cm = pd.DataFrame(cm)
        cm.columns = ["Signal(true)", "Background(true)"]
        cm.index = ["Signal(pred)", "Background(pred)"]

        # Use matplotlib to create the heatmap
        cax = ax.matshow(cm, cmap="Blues")

        # Add color bar for reference
        plt.colorbar(cax)

        ax.set_xticks(
            np.arange(cm.shape[1])
        )  # Set tick positions for the number of columns
        ax.set_yticks(
            np.arange(cm.shape[0])
        )  # Set tick positions for the number of rows
        ax.set_xticklabels(cm.columns, rotation=45)
        ax.set_yticklabels(cm.index)

        # Annotate each cell with the numeric value
        for (i, j), val in np.ndenumerate(cm.values):
            ax.text(
                j, i, f"{val:.2f}", ha="center", va="center", color="black", fontsize=16
            )

        plt.tight_layout()
        plt.savefig(trainer.log_dir + "_cm.png", dpi=300)

        # --------------------------------------------------------------------------------
        # Confusion Matrix
        # --------------------------------------------------------------------------------

        fig_1 = plt.figure(figsize=(12, 10))
        _ = plt.hist(
            predictions[np.where(targets == 1)[0]],
            # predictions[targets==1,0],
            bins=100,
            alpha=0.3,
            color="blue",
            label="signal",
            density=True,
        )
        _ = plt.hist(
            predictions[np.where(targets == 0)[0]],
            # predictions[targets==0,0],
            bins=100,
            alpha=0.3,
            color="red",
            label="background",
            density=True,
        )
        plt.semilogy()
        plt.xlabel("GNN Score", fontsize=16)
        plt.ylabel("Density", fontsize=16)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.legend(loc="upper right", fontsize=16)
        plt.tight_layout()
        plt.savefig(trainer.log_dir + "_output.png", dpi=300)

        # free up the memory
        module.test_predictions.clear()
        module.test_targets.clear()
