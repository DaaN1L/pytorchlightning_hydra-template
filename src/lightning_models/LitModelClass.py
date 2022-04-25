from pathlib import Path

import numpy as np
from omegaconf import DictConfig
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import torchmetrics
from torchmetrics.functional import accuracy, recall, precision

from src.utils.get_dataset import get_dataset
from src.utils.get_model import get_model
from src.utils.utils import load_obj


class LitModelClass(pl.LightningModule):
    def __init__(self, hparams: DictConfig):
        super().__init__()
        self.params = hparams

        self.model, self.freeze_up_to = get_model(self.params)
        self.loss = load_obj(self.params.loss.class_name)()

        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()

        self.test_acc = torchmetrics.Accuracy()
        self.test_prec = torchmetrics.Precision()
        self.test_rec = torchmetrics.Recall()

    def forward(self, x):
        return self.model(x)

    def prepare_data(self) -> None:
        datasets = get_dataset(self.params)

        self.train_dataset = datasets["train"]
        self.val_dataset = datasets["val"]
        self.test_dataset = datasets["test"]

    def configure_optimizers(self):
        optimizer = load_obj(self.params.optimizer.class_name)(
            self.model.parameters(), **self.params.optimizer.params
        )
        scheduler = load_obj(self.params.scheduler.class_name)(
            optimizer, **self.params.scheduler.params
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": self.params.training.metric,
        }

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.squeeze()
        y_hat = self(x)
        loss = self.loss(y_hat, y)  # , weight=self.class_weights

        self.train_acc(torch.softmax(y_hat, dim=1), y)

        self.log_dict(
            {"loss": loss, "train_acc": self.train_acc},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.squeeze()
        y_pred = self(x)

        val_loss = self.loss(y_pred, y)
        self.val_acc(torch.softmax(y_pred, dim=1), y)

        self.log_dict(
            {"val_loss": val_loss, "val_acc": self.val_acc},
            on_step=False,
            on_epoch=True,
        )

    def train_dataloader(self):
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.params.data.batch_size,
            num_workers=self.params.data.num_workers,
            shuffle=True,
        )

        # init class weights if necessary
        # if self.params.loss.params.class_weight:
        #     self.class_weights = compute_weights(train_dataloader)
        # else:
        #     self.class_weights = None
        self.class_weights = None
        return train_dataloader

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.params.data.batch_size,
            num_workers=self.params.data.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.params.data.batch_size,
            num_workers=self.params.data.num_workers,
            shuffle=False,
        )

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = torch.softmax(self(x))

        self.test_acc(y_pred, y)
        self.test_prec(y_pred, y)
        self.test_rec(y_pred, y)

        return {"y": y, "y_pred": y_pred}

    def test_epoch_end(self, outputs):
        suffix = Path(self.test_dataloader.dataloader.dataset.csv_path).stem
        self.log_dict(
            {
                f"test_acc_0.5_{suffix}": self.test_acc,
                f"test_prec_0.5_{suffix}": self.test_prec,
                f"test_rec_0.5_{suffix}": self.test_rec,
            }
        )

        y = torch.cat([x["y"] for x in outputs], dim=0).squeeze()
        y_pred = torch.cat([x["y_pred"] for x in outputs], dim=0).squeeze()

        best_acc, best_thr = -1, -1
        for thr in np.arange(0.01, 1, 0.01):
            y_pred_thr = torch.ge(y_pred, thr).int()
            acc = accuracy(y_pred_thr, y)
            if acc > best_acc:
                best_acc = acc
                best_thr = thr

        y_pred_thr = torch.ge(y_pred, best_thr).int()
        best_prec = precision(y_pred_thr, y)[1]
        best_rec = recall(y_pred_thr, y)[1]

        self.log_dict(
            {
                f"best_acc_{suffix}": best_acc,
                f"best_thr_{suffix}": best_thr,
                f"best_prec_{suffix}": best_prec,
                f"best_rec_{suffix}": best_rec,
            }
        )
