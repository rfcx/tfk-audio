import tensorflow as tf

tf.config.run_functions_eagerly(True)

def masked_recall_at_precision(y_true: tf.Tensor, y_pred: tf.Tensor, precision: tf.float32 = 0.95):
    ''' Wraps tf.keras.metrics.RecallAtPrecision metric with a masking operation
    
    The masking operation sets the loss to zero for elements with unknown labels indicated by -1
    
    Args:
        y_true: true tensor
        y_pred: predicted tensor
        precision: desired precision to evaluate recall at
    Returns:
        Score
    '''
    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.convert_to_tensor(y_pred)
    return tf.keras.metrics.RecallAtPrecision(precision=precision)(
        tf.where(y_true==-1, tf.zeros_like(y_true), y_true),
        tf.multiply(y_pred, tf.cast(tf.logical_not(y_true==-1), tf.float32))
    )

def masked_precision_at_recall(y_true: tf.Tensor, y_pred: tf.Tensor, recall: tf.float32 = 0.95):
    ''' Wraps tf.keras.metrics.PrecisionAtRecall metric with a masking operation
    
    The masking operation sets the loss to zero for elements with unknown labels indicated by -1
    
    Args:
        y_true: true tensor
        y_pred: predicted tensor
        recall: desired recall to evaluate precision at
    Returns:
        Score
    '''
    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.convert_to_tensor(y_pred)
    return tf.keras.metrics.PrecisionAtRecall(recall=recall)(
        tf.where(y_true==-1, tf.zeros_like(y_true), y_true),
        tf.multiply(y_pred, tf.cast(tf.logical_not(y_true==-1), tf.float32))
    )