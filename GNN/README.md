# Higgs CP Structure Identification using Graph Neural Network

## Prerequisites

Before you begin, ensure you have git installed on your machine to clone this repository. If git is not installed, you can download it from [Git's official site](https://git-scm.com/downloads).

## Installation

Follow these steps to set up your environment and start analyzing the Higgs boson's CP structure.

### Step 1: Clone the Repository

Clone this repository to your local machine using the following command:

```bash
git clone https://github.com/AHamamd150/Conditional_GNN.git
cd Conditional_GNN/GNN
```

### Step 2: Install Conda

If you do not have Miniconda or Anaconda installed, download and install it from [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/individual) respectively.

### Step 3: Set Up Your Environment

This project relies on several dependencies listed in `environment.yml`, including libraries such as NumPy, Pandas, Matplotlib, tqdm, h5py, scikit-learn, PyTorch, PyTorch Geometric, PyTorch Lightning, and Torchmetrics.

To install all dependencies at once and create a Conda environment named `higgscp`, run the following command in your terminal:

```bash
conda env create -f environment.yml
```

### Step 4: Activate the Environment
```bash
conda activate higgscp
```

## Usage

### Get the Data

### Training the Model

You can train the model using the `run.sh` script provided in the repository. This script supports running the training process.

To see how to use the script, you can type:

```bash
./run.sh -h
```
For example, to train the multi-modal GNN model, you can use the following command:

```bash
./run.sh --mode "train"
```
This command will create an HDF5 file named `inputgraphs.hdf5` in the `\files` subdirectory containing the signal and background data, which will be used to train the model. The training results, including the saved model, hyperparameters, and training progress, will be stored in a directory named `results`.

### Configuration
The training (and testing) hyperparameters, such as the number of epochs, learning rate, and size of the network, are stored in the `config.yaml` file located in the `config/` directory. You can override any of these parameters by modifying this file before running the training or testing commands.


### Testing the Trained Model
To test a trained model, you need to provide the path to the trained model's checkpoint file to the `run.sh` script. For example, to test a model trained on angle `0` can be something like the following line, where you need to adjust the `ckpt_path` where the trained model is:

```bash
./run.sh --mode "test" --ckpt_path "results/version_0/checkpoints/epoch=0-step=109.ckpt" --h5_file "files/inputgraphs_0.000000.hdf5"
```
This command will perform the testing on the specified model and data file `inputgraphs_0.000000.hdf5`, where `0.000000` is the angle. The results, including **ROC** and the **Confusion Matrix**, will be saved under the `results` directory.



