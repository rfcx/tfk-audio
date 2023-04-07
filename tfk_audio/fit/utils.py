import tensorflow as tf


class MaskedMetric(tf.keras.metrics.Metric):
    ''' Ignores masked elements from a metric function
    '''
    def __init__(self, metric, name='masked_metric', mask_val=-1.0, **kwargs):
        super(MaskedMetric, self).__init__(name=name, **kwargs)
        self.metric = metric
        self.mask_val = mask_val
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        masked_y_true, masked_y_pred, mask = get_masked_arrays(y_true, y_pred, self.mask_val)
        self.metric.update_state(masked_y_true, masked_y_pred, sample_weight=sample_weight)
        self.total.assign_add(tf.reduce_sum(masked_y_pred))
        self.count.assign_add(tf.reduce_sum(mask))

    def result(self):
        return self.metric.result()

    def reset_state(self):
        self.total.assign(0)
        self.count.assign(0)
        self.metric.reset_state()
        
        
class MaskedLoss(tf.keras.losses.Loss):
    ''' Ignores masked elements from a loss function
    '''
    def __init__(self, loss_fn, name='masked_loss', mask_val=-1.0, **kwargs):
        super(MaskedLoss, self).__init__(name=name, **kwargs)
        self.loss_fn = loss_fn
        self.mask_val = mask_val
    
    def call(self, y_true, y_pred, sample_weight=None):
        masked_y_true, masked_y_pred, mask = get_masked_arrays(y_true, y_pred, self.mask_val)
        masked_loss = self.loss_fn(masked_y_true, masked_y_pred, sample_weight=sample_weight)
        return masked_loss / tf.reduce_sum(mask) # divide by number of non-masked elements, be careful if self.loss_fn does
    
    
def get_masked_arrays(y_true, y_pred, mask_val):
    ''' Sets elements where y_true = mask_val to 0 in y_true and y_pred
    '''
    mask = tf.not_equal(y_true, mask_val)
    mask = tf.cast(mask, dtype=y_pred.dtype)
    masked_y_true = y_true * mask
    masked_y_pred = y_pred * mask
    return masked_y_true, masked_y_pred, mask
    
    
    
    
    
    
    
    
    