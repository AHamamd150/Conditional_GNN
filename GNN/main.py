# Torch imports
import torch

# PyTorch Lightning imports
from lightning.pytorch.cli import LightningCLI

# Framework imports
from models.models import HomoGNN
from data.data_handling import HiggsHomoDataModule


def cli_main():
    cli = LightningCLI(HomoGNN, HiggsHomoDataModule)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    cli_main()
