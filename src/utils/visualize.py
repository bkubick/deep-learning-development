import tensorflow as tf
from tensorflow.keras.utils import plot_model


def visualize_model(model: tf.keras.Model):
    """ Visualizes the model structure (layers and their neurons).

        NOTE: Only works if pydot and graphviz are installed, and on a Jupyter notebook.
    
        Args:
            model: The model to visualize.
    """
    return plot_model(model, show_shapes=True, show_layer_names=True)
