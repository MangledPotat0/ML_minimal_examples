# -*- coding: utf-8 -*-
"""
Reusable blocks for generating figures
"""

# Built-in module imports
from typing import List

# 3rd party module imports
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
import seaborn as sns
from sklearn.metrics import confusion_matrix as cm

def confusion_matrix(y_true: ndarray, y_pred: ndarray, labels: List[str],
                     figname: str = None):
    """
    Generates and renders confusion matrix.

    Args:
        y_true (ndarray): Ground truth labels in dense vector form.
        y_pred (ndarray): Prediction labels in dense vector form.
        labels List[str]: Class label strings
    """
    cmat = cm(y_true, y_pred)
    ax = sns.heatmap(cmat, annot=True, xticklabels=labels, yticklabels=labels)
    plt.ylabel("Ground Truth")
    plt.xlabel("Predictions")
    ax.xaxis.tick_top()
    plt.savefig(f"{figname}_confusion_matrix.png")
    plt.close()

# EOF
