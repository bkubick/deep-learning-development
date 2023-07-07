from __future__ import annotations

import itertools
import math
import random
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import tensorflow as tf

Array = Union[tf.Tensor, np.ndarray]

# ============================== GENERAL PLOTTING ==============================


def plot_true_versus_predicted(y_true: Array, y_predict: Array, figsize: Optional[Tuple[int, int]] = (10, 7)):
    """ Plots the actual true values against the predicted values.
        Note that better predictions have a slope closer to 1.

        NOTE: This is a very simple plot, and is not very useful for more complex models.
        NOTE: This only works for regression problems.

        Args:
            y_true (Array): The true values.
            y_predict (Array): The predicted values.
            figsize (Optional[Tuple[int, int]]): The size of the figure.
    """
    plt.figure(figsize=figsize)

    plt.title('Actual Value vs. Predicted Value')
    plt.xlabel('Actual Value')
    plt.ylabel('Predicted Value')
    plt.scatter(y_true, y_predict)

    plt.show()

    
def plot_history(history: tf.keras.callbacks.History):
    """ Plots the loss versus the epochs.
    
        Args:
            history (tf.keras.callbacks.History): The history object returned by the fit method.
    """
    pd.DataFrame(history.history).plot()
    plt.ylabel('Metrics')
    plt.xlabel('Epochs')
    plt.title('Metrics vs. Epochs')
    plt.show()


def plot_learning_rate_versus_loss(learning_rates: List[float], losses: List[float]):
    """ Plots the loss versus the learning rate for each epoch.

        Args:
            learning_rates (List[float]): The learning rates.
            losses (List[float]): The losses.
    """
    plt.figure(figsize=(10, 7))
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('Learning Rate vs. Loss')
    plt.plot(learning_rates, losses)
    plt.xscale('log')
    plt.show()


# ============================== CONFUSION MATRIX PLOTTING ==============================


def plot_confusion_matrix(y_true: Array,
                          y_pred: Array,
                          label_text_size: int = 20,
                          cell_text_size: int = 10,
                          classes: Optional[List[str]] = None,
                          figsize: Optional[Tuple[int, int]] = (15, 15)):
    """ Plots a confusion matrix using Seaborn's heatmap. 
    
        Args:
            y_true (Array): The true values.
            y_pred (Array): The predicted values.
            label_text_size (int): The size of the labels.
            cell_text_size (int): The size of the text in each cell.
            classes (Optional[List[str]]): The class labels of each category.
            figsize (Optional[Tuple[int, int]]): The size of the figure.
    """
    # Create the confusion matrix
    cm = confusion_matrix(y_true, tf.round(y_pred))

    cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize our confusion matrix
    n_classes = cm.shape[0]

    # Prettifying it
    fig, ax = plt.subplots(figsize=figsize)
    # Create a matrix plot
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    # Create classes
    if classes:
        labels = classes
    else:
        labels = np.arange(cm.shape[0])

    # Label the axes
    ax.set(title='Confusion Matrix',
          xlabel='Predicted Label',
          ylabel='True Label',
          xticks=np.arange(n_classes),
          yticks=np.arange(n_classes),
          xticklabels=labels,
          yticklabels=labels)

    # Make Labels bigger
    ax.yaxis.label.set_size(label_text_size)
    ax.xaxis.label.set_size(label_text_size)
    ax.title.set_size(label_text_size)

    # Set the threshold
    threshold = (cm.max() + cm.min()) / 2

    # Plot the text on each cell
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f'{cm[i, j]} ({cm[i, j]:.1f}%)',
                 horizontalalignment='center',
                 color='white' if cm[i,j] > threshold else 'black',
                 size=cell_text_size)
    
    fig.show()


# ============================== IMAGE PLOTTING ==============================

def plot_image(index: int, images: Array, labels: Array, class_names: List[str], black_and_white: bool = False):
    """ Plots an image from the dataset.

        Args:
            index (int): The index of the image to plot.
            images (Array): The data to plot.
            labels (Array): The labels of the data.
            class_names (List[str]): The names of the classes.
            black_and_white (bool): Whether to plot the image in black and white.
    """
    cmap = None
    if black_and_white:
        cmap = plt.cm.binary

    plt.imshow(images[index], cmap=cmap)
    plt.title(class_names[labels[index]])
    plt.show()


def plot_images(indexes: List[int],
                images: Array,
                labels: Array,
                class_names: List[str],
                black_and_white: bool = False):
    """ Plots each image from the dataset at each corresponding index.

        Args:
            indexes (List[int]): The index of the image to plot.
            images (Array): The data to plot.
            labels (Array): The labels of the data.
            class_names (List[str]): The names of the classes.
            black_and_white (bool): Whether to plot the image in black and white.
    """
    assert len(indexes) <= 4, 'Cannot plot more than 4 images at a time.'

    cmap = None
    if black_and_white:
        cmap = plt.cm.binary

    total_images_per_row: int = 4
    total_rows: int = math.ceil(len(indexes) / total_images_per_row)

    fig, axes = plt.subplots(total_rows, min([total_images_per_row, len(indexes)]), figsize=(15, 7))
    for i, fig_index in enumerate(indexes):
        axes[i].imshow(images[fig_index], cmap=cmap)
        axes[i].set_title(class_names[labels[fig_index]])

    fig.show()


def plot_random_image_label_and_prediction(images: Array,
                                           true_labels: Array,
                                           pred_probabilities: Array,
                                           class_names: List[str],
                                           black_and_white: bool = False):
    """ Plots a random image from the dataset.

        NOTE: images and labels must be the same length.
        NOTE: class_names must be the same length as the number of classes in the dataset.

        Args:
            images (Array): The data to plot.
            true_labels (Array): The true labels of the data.
            pred_probabilities (Array): The predicted probabilities of the data.
            class_names (List[str]): The names of the classes.
            black_and_white (bool): Whether to plot the image in black and white.
    """
    index_of_choice = random.randint(0, len(images))

    target_image = images[index_of_choice]

    pred_label = class_names[pred_probabilities[index_of_choice].argmax()]
    true_label = class_names[true_labels[index_of_choice]]

    cmap = None
    if black_and_white:
        cmap = plt.cm.binary

    plt.imshow(target_image, cmap=cmap)

    x_label_color = 'green' if pred_label == true_label else 'red'
    plt.xlabel(f'Pred: {pred_label}  True: {true_label}', color=x_label_color)

    plt.show