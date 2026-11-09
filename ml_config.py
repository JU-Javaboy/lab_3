"""Shared configuration. All inputs are teaching artifacts, not clinical tools."""
from pathlib import Path
ROOT = Path(__file__).resolve().parent
FEATURES = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness']
TARGET = "malignant"  # 1=malignant, 0=benign; adapted from sklearn's labeling
REQUIRED = ["sample_id", *FEATURES, TARGET]
