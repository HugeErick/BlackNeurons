import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

def plot_training_loss(losses, output_path, title='Training Loss'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure()
    plt.plot(losses)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(output_path)
    plt.close()

def plot_confusion_matrix(cm, output_path, labels=None, title='Confusion Matrix'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.savefig(output_path)
    plt.close()
