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
    # TODO: Move this to a nice function
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


def create_tensorboard_callback(dir_name: str, experiment_name: str) -> tf.keras.callbacks.TensorBoard:
    """ Creates a TensorBoard callback.
    
        Args:
            dir_name: The directory name.
            experiment_name: The experiment name.

        Returns:
            The TensorBoard callback.
    """
    log_dir = f"{dir_name}/{experiment_name}/{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    print('Saving TensorBoard log files to: ', log_dir)
    
    return tensorboard_callback
