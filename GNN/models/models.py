# torch imports
import torch
import torch.nn.functional as F

# PyTorch Geometric imports
import torch_geometric

# PyTorch lightning imports
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT

# TorchMetrics
from torchmetrics import Accuracy, AUROC

ACTIVATION = "sigmoid"


class HybridGNN(LightningModule):
    def __init__(
        self, in_feat=6, node_feat=8, h_feat=16, dropout=0.1, n_heads=8, num_classes=2
    ):
        super(HybridGNN, self).__init__()

        self.gnn = torch_geometric.nn.GAT(
            in_channels=node_feat,
            hidden_channels=h_feat,
            num_layers=3,
            out_channels=h_feat // 2,
            heads=n_heads,
            act="leaky_relu",
            jk="cat",
            norm="BatchNorm",
            v2=True,
            dropout=dropout,
        )

        self.lin1 = torch.nn.Sequential(
            torch.nn.Linear(in_feat, h_feat),
            torch.nn.BatchNorm1d(h_feat),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(h_feat, h_feat),
            torch.nn.BatchNorm1d(h_feat),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(h_feat, h_feat),
            torch.nn.LeakyReLU(),
        )

        self.lin2 = torch.nn.Sequential(
            torch.nn.Linear(h_feat + 1, h_feat // 2),
            torch.nn.LeakyReLU(),
        )

        self.lin3 = torch.nn.Sequential(
            torch.nn.Linear(h_feat + 1, h_feat // 2),
            torch.nn.LeakyReLU(),
        )

        self.clf = torch.nn.Linear(h_feat, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        feats: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        x = self.gnn(x, edge_index, edge_attr)
        x1 = torch_geometric.nn.global_max_pool(x, batch)
        x2 = torch_geometric.nn.global_mean_pool(x, batch)
        x = torch.concat([x1, x2], dim=1)

        h = self.lin1(feats)

        h1 = self.lin2(torch.concat([x, alpha], dim=1))
        h2 = self.lin3(torch.concat([h, alpha], dim=1))

        return self.clf(torch.concat([h1, h2], dim=1))


class HomoGNN(LightningModule):
    def __init__(
        self,
        in_feat=94,
        node_feat=8,
        h_feat=32,
        n_heads=8,
        num_classes=2,
        dropout=0.1,
        batch_size=32,
        learning_rate=0.0001,
        angle="",
    ) -> None:
        super().__init__()
        self.in_feat = in_feat
        self.node_feat = node_feat
        self.h_feat = h_feat
        self.n_heads = n_heads
        self.dropout = dropout
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.lr = learning_rate
        self.angle = angle

        # Saving hyperparameters
        self.save_hyperparameters()

        self.model = HybridGNN(
            in_feat=self.in_feat,
            node_feat=self.node_feat,
            h_feat=self.h_feat,
            n_heads=self.n_heads,
            dropout=self.dropout,
            num_classes=self.num_classes,
        )

        # class weights to penalize the large false negatives
        self.negative_class_weight = 2

        # initialize metric
        self.train_metric = Accuracy(task="binary")
        self.val_metric = Accuracy(task="binary")
        self.test_metric = Accuracy(task="binary")
        self.test_auc = AUROC(task="binary")
        if ACTIVATION == "softmax":
            self.train_metric = Accuracy(
                task="multiclass", num_classes=self.num_classes
            )
            self.val_metric = Accuracy(task="multiclass", num_classes=self.num_classes)
            self.test_metric = Accuracy(task="multiclass", num_classes=self.num_classes)
            self.test_auc = AUROC(task="multiclass", num_classes=self.num_classes)

        # for predictions
        self.test_predictions = []
        self.test_targets = []

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=0.001
        )
        return optimizer

    def training_step(self, data, batch_idx) -> STEP_OUTPUT:
        graphs, feats, alpha = data[0], data[1], data[2]
        labels = graphs.y
        if ACTIVATION == "sigmoid":
            labels = labels.float()
        linear_out = self.model(
            graphs.x, graphs.edge_index, graphs.edge_attr, graphs.batch, feats, alpha
        )
        self.class_weights = labels + (labels == 0) * self.negative_class_weight
        pred, loss = None, None
        if ACTIVATION == "softmax":
            self.class_weights = torch.tensor([self.negative_class_weight, 1.0]).to(
                next(self.model.parameters()).device
            )
            pred = torch.nn.functional.softmax(linear_out, dim=1)
            loss = torch.nn.functional.cross_entropy(
                pred, labels, weight=self.class_weights
            )
        elif ACTIVATION == "sigmoid":
            pred = torch.nn.functional.sigmoid(linear_out).flatten()
            loss = torch.nn.functional.binary_cross_entropy(
                pred, labels, weight=self.class_weights
            )

        self.log("train_loss", loss, batch_size=self.batch_size, prog_bar=True)
        self.log(
            "train_acc",
            self.train_metric(pred, labels),
            batch_size=self.batch_size,
            prog_bar=True,
        )
        return loss

    def validation_step(self, data, batch_idx) -> STEP_OUTPUT:
        graphs, feats, alpha = data[0], data[1], data[2]
        labels = graphs.y
        if ACTIVATION == "sigmoid":
            labels = labels.float()
        linear_out = self.model(
            graphs.x, graphs.edge_index, graphs.edge_attr, graphs.batch, feats, alpha
        )

        val_pred, val_loss = None, None
        if ACTIVATION == "softmax":
            val_pred = torch.nn.functional.softmax(linear_out, dim=1)
            val_loss = torch.nn.functional.cross_entropy(val_pred, labels)
        elif ACTIVATION == "sigmoid":
            val_pred = torch.nn.functional.sigmoid(linear_out).flatten()
            val_loss = torch.nn.functional.binary_cross_entropy(val_pred, labels)

        self.log(
            "val_loss",
            val_loss,
            sync_dist=True,
            batch_size=self.batch_size,
            prog_bar=True,
        )
        self.log(
            "val_acc",
            self.val_metric(val_pred, labels),
            sync_dist=True,
            batch_size=self.batch_size,
            prog_bar=True,
        )

    def test_step(self, data, batch_idx) -> STEP_OUTPUT:
        graphs, feats, alpha = data[0], data[1], data[2]
        labels = graphs.y
        if ACTIVATION == "sigmoid":
            labels = labels.float()
        linear_out = self.model(
            graphs.x, graphs.edge_index, graphs.edge_attr, graphs.batch, feats, alpha
        )
        test_pred = None
        if ACTIVATION == "softmax":
            test_pred = torch.nn.functional.softmax(linear_out, dim=1)
        elif ACTIVATION == "sigmoid":
            test_pred = torch.nn.functional.sigmoid(linear_out).flatten()

        self.log(
            "test_acc",
            self.test_metric(test_pred, labels),
            batch_size=self.batch_size,
            prog_bar=True,
        )
        self.log(
            "test_auc",
            self.test_auc(test_pred, labels),
            batch_size=self.batch_size,
            prog_bar=True,
        )

        # for predictions
        self.test_predictions.extend(test_pred.detach().cpu().numpy())
        self.test_targets.extend(labels.cpu().numpy())
