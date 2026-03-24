# CS515 HW1b — Transfer Learning & Knowledge Distillation

CIFAR-10 classification experiments using transfer learning, knowledge distillation, and label smoothing.

## Project Structure

```
├── main.py              # Main script — runs all experiments
├── parameters.py        # Argument parsing and config dataclasses
├── train.py             # Training functions (standard, distillation, soft labels)
├── test.py              # Evaluation and FLOPs counting
├── visualize.py         # Plotting (curves, t-SNE, confusion matrix, FLOPs comparison)
├── models/
│   ├── __init__.py
│   ├── simple_cnn.py    # SimpleCNN baseline
│   ├── resnet.py        # ResNet-18 (transfer learning + teacher)
│   └── mobilenet.py     # MobileNetV2 (student)
└── plots/               # Generated visualizations
```

## Setup

```bash
pip install torch torchvision matplotlib scikit-learn ptflops
```

## Usage

```bash
python main.py --epochs 10
```

Device is auto-detected (CUDA → MPS → CPU). Override with `--device cpu`.

All available arguments:

| Argument        | Default | Description                    |
|-----------------|---------|--------------------------------|
| `--lr`          | 0.001   | Learning rate                  |
| `--epochs`      | 20      | Number of epochs               |
| `--batch_size`  | 64      | Batch size                     |
| `--device`      | auto    | Device (cuda / mps / cpu)      |
| `--temperature` | 3.0     | Distillation temperature       |
| `--alpha`       | 0.2     | Hard label weight              |
| `--epsilon`     | 0.2     | Label smoothing factor         |

## Experiments

1. **Transfer Learning** — ResNet-18 pretrained on ImageNet, adapted for CIFAR-10 (Option 1: resize to 224, Option 2: modify conv1)
2. **SimpleCNN Baseline** — Small CNN trained from scratch
3. **ResNet from Scratch** — With and without label smoothing
4. **Knowledge Distillation** — SimpleCNN trained with ResNet teacher
5. **Soft Label Training** — MobileNetV2 trained with teacher's class probabilities
