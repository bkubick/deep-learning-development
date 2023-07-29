import datetime as dt
import os
import pathlib

import numpy as np
import tensorflow as tf


def summarize_image_directory(data_directory: pathlib.Path):
    """ Summarizes the number of images in each directory of the data directory.

        Args:
            data_directory: The directory containing the data.
    """
    # Lets look at the number of files in the test and train sets
    for dirpath, dirnames, filenames in os.walk(data_directory):
        images = [file for file in filenames if file.endswith('jpg') or file.endswith('jpeg') or file.endswith('png')]
        if images:
            print(f'Directory: {dirpath} Total Images: {len(images)}')


def get_classnames_from_directory(data_directory: pathlib.Path) -> np.ndarray:
    """ Gets the class names from the data directory.
    
        Args:
            data_directory: The directory containing the data.
        
        Returns:
            The class names.
    """
    all_class_names = [
        item.name for item in data_directory.iterdir() if item.is_dir() and not item.name.startswith('.')
    ]
    class_names = np.array(sorted(all_class_names))
    return class_names


def load_and_prep_image(filename: str, image_size: int = 224, scale: bool = True) -> tf.Tensor:
    """ Loads and prepares an image for the model.
    
        Args:
            filename: The filename of the image.
            image_size: The size of the image.
            scale: Whether to scale the image or not.

        Returns:
            The prepped image.
    """
    # Read in the image
    image = tf.io.read_file(filename)
    image = tf.io.decode_image(image, channels=3)
    image = tf.image.resize(image, [image_size, image_size])

    # Scale Image to get all between 0 & 1 (not always required)
    if scale:
        image = image / 255.

    return image
