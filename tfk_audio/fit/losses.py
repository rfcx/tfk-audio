import tensorflow as tf

def masked_binary_crossentropy(y_true: tf.Tensor, y_pred: tf.Tensor):
    ''' Wraps binary crossentropy loss function with a masking operation
    
    The masking operation sets the loss to zero for elements with unknown labels indicated by -1
    
    Args:
        y_true: true tensor
        y_pred: predicted tensor
    Returns:
        Loss value
    '''
    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.convert_to_tensor(y_pred)
    return tf.keras.losses.BinaryCrossentropy()(tf.where(y_true==-1, tf.zeros_like(y_true), y_true),
                                                tf.multiply(y_pred, tf.cast(tf.logical_not(y_true==-1), tf.float32)))
