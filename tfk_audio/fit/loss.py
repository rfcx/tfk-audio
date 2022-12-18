import tensorflow as tf

def mask_loss(y_true: tf.Tensor, y_pred: tf.Tensor, loss_fn: tf.keras.losses.Loss):
    ''' Wraps a loss function with a masking operation
    
    The masking operation sets the loss to zero for elements with unknown labels indicated by -1
    
    Args:
        y_true: true tensor
        y_pred: predicted tensor
        loss_fn: tensorfow.keras-compatible loss function
    Returns:
        A loss value computed by loss_fn
    '''
    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.convert_to_tensor(y_pred)
    return loss_fn(tf.where(y_true==-1, tf.zeros_like(y_true), y_true),
                   tf.multiply(y_pred, tf.cast(tf.logical_not(y_true==-1), tf.float32)))