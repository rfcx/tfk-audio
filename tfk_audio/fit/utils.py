import tensorflow as tf

class masked_metric_or_loss():
    ''' Wraps a metric/loss function and sets any input values where y_true==mask_val to 0
    '''
    def __init__(self,
                 fn,
                 mask_val = -1):
        '''
        Args:
            fn:            metric/loss function taking arguments y_true, y_pred, and sample_weights
            mask_val:      elements where the target array equals this value will be masked
        '''
        self.fn = fn
        self.mask_val = -1
        
    def __call__(self, y, p, sample_weights=None):
        y = tf.convert_to_tensor(y)
        p = tf.convert_to_tensor(p)
        y_masked = tf.where(y==-1, tf.zeros_like(y), y)
        p_masked = tf.multiply(p, tf.cast(tf.logical_not(y==-1), tf.float32))
        return self.fn(y_masked, p_masked, sample_weights)

    