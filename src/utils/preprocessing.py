from typing import Callable, Tuple

import tensorflow as tf


def generate_preprocess_image_function(img_shape: int = 224) -> Callable[[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]:
    """ Generates the preprocessing function for the given image shape.

        - Convert image dtype from `uint8` to `float32`
        - Reshapes image to (img_shape, img_shape, color_channels)

        Args:
            img_shape (int): The size the img should be shaped to
        
        Returns:
            The preprocessing function.
    """
    def preprocess_image(image: tf.Tensor, label: tf.Tensor):
        """ Preprocess the given image performing the following:

            - Convert image dtype from `uint8` to `float32`
            - Reshapes image to (img_shape, img_shape, color_channels)

            Args:
            image (tf.Tensor): Image to preprocess
            label (tf.Tensor): Label for the corresponding image
        """
        image = tf.image.resize(image, [img_shape, img_shape])
        return tf.cast(image, tf.float32), label

    return preprocess_image
