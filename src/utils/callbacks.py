import datetime as dt
from typing import Optional

import tensorflow as tf


def generate_tensorboard_callback(dir_name: str, experiment_name: str) -> tf.keras.callbacks.TensorBoard:
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


def generate_checkpoint_callback(checkpoint_path: str,
                                 monitor: Optional[str] = None,
                                 best_only: bool = True) -> tf.keras.callbacks.ModelCheckpoint:
    """ Generates a checkpoint callback.

        Args:
            checkpoint_path (str): The path to save the checkpoint to.
            monitor (Optional[str]): The metric to monitor.
            best_only (bool): Whether to save only the best model.
        
        Returns:
            The checkpoint callback.
    """
    if monitor is None:
        monitor = 'val_accuracy'

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor=monitor,
        save_weights_only=True,
        save_best_only=best_only,
        save_freq='epoch',
        verbose=1)

    return checkpoint


def generate_csv_logger_callback(filename: str, logs_dir: Optional[str] = None) -> tf.keras.callbacks.CSVLogger:
    """ Generates a CSV logger callback.
    
        Args:
            filename (str): The filename of the CSV logger.
            logs_dir (Optional[str]): The directory to save the CSV logger to.
        
        Returns:
            The CSV logger callback.
    """
    if logs_dir is None:
        logs_dir = 'logs/csv'

    logger = tf.keras.callbacks.CSVLogger(f'{logs_dir}/{filename}')
    return logger
