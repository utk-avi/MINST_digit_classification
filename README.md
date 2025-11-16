# MNIST Digit Classification

A simple and clean PyTorch-based implementation for classifying handwritten digits from the MNIST dataset.

## 📋 Table of Contents

* Project Overview
* Dataset
* Getting Started
* Prerequisites
* Installation
* Usage
* Project Structure
* Training & Evaluation
* Results
* Contributing
* License

## Project Overview

This repository provides a straightforward implementation of a neural network using PyTorch to classify images of handwritten digits (0–9). It is designed to be clean, minimal, and easy to understand, making it ideal for learning or experimenting.

## Dataset

This project uses the standard MNIST dataset.
PyTorch automatically handles:

* downloading the dataset
* extracting it
* storing it in the `root` directory you specify

You don’t need to manually download anything.
In the code:

* `root` = folder where MNIST will be saved
* `train=True/False` selects train or test set
* `transform` applies operations like converting to tensors

## Getting Started

### Prerequisites

* Python 3.8+
* PyTorch
* torchvision

### Installation

Clone the repo:

```bash
git clone https://github.com/utk-avi/MINST_digit_classification.git
cd MNST_digit_classification
```

(Optional) Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install torch torchvision
```

## Usage

Run the main training script:

```bash
python code.py
```

The dataset will automatically download into the directory you set as `root`.
Training and evaluation logs will be printed in the terminal.

## Project Structure

```
MNST_digit_classification/
│
├─ code.py                  # Main script: loads data, builds model, trains & tests
├─ model.py                 # (if present) model architecture
├─ utils.py                 # (if present) helper functions
├─ data/                    # MNIST dataset stored here automatically
├─ README.md
└─ requirements.txt
```

## Training & Evaluation

* Uses CrossEntropyLoss and optimizers (Adam/SGD)
* Tracks loss and accuracy across epochs
* Evaluates on the MNIST test dataset after training
  With a simple CNN, you can expect ~98% accuracy on MNIST.

## Results

Typical results include high accuracy and low loss on the test set.
You can improve performance by using deeper models or tuning hyperparameters.

## Contributing

You can contribute by improving architecture, adding more datasets (e.g., Fashion-MNIST), visualisations, or configurable parameters.

## License

This project is licensed under the MIT License.
