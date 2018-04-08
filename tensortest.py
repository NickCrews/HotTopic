import tensorflow as tf


_FLOATX = 'float32'
_EPSILON = 1e-7
_IMAGE_DATA_FORMAT = 'channels_last'


def epsilon():
    """Returns the value of the fuzz factor used in numeric expressions.
    # Returns
        A float.
    # Example
    ```python
        >>> keras.backend.epsilon()
        1e-07
    ```
    """
    return _EPSILON

def clip(tensor):
    return tf.clip_by_value(tensor, clip_value_min=.0, clip_value_max=1.)

def mean(x, axis=None, keepdims=False):
    """Mean of a tensor, alongside the specified axis.
    # Arguments
        x: A tensor or variable.
        axis: A list of integer. Axes to compute the mean.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1 for each entry in `axis`. If `keepdims` is `True`,
            the reduced dimensions are retained with length 1.
    # Returns
        A tensor with the mean of elements of `x`.
    """
    if x.dtype.base_dtype == tf.bool:
        x = tf.cast(x, floatx())
    return tf.reduce_mean(x, axis, keepdims)

shape = (3,10,5,1)
his = clip(tf.random_normal(shape, mean=.9, stddev=.1))
los = clip(tf.random_normal(shape, mean=.1, stddev=.1))
choices1 = tf.random_uniform(shape, minval=0., maxval=1.)
choices2 = tf.random_uniform(shape, minval=0., maxval=1.)
yhat_rand = tf.where(choices2<.5, los, his)
y_rand = tf.where(choices1<.5, los, his)


# y = tf.where()
# y = tf.random_uniform((3,4), minval=0., maxval=1., name='y')
# yhat = tf.random_uniform((3,4), minval=0., maxval=1., name='y')

def _to_tensor(x, dtype):
    """Convert the input `x` to a tensor of type `dtype`.
    # Arguments
        x: An object to be converted (numpy array, list, tensors).
        dtype: The destination type.
    # Returns
        A tensor.
    """
    return tf.convert_to_tensor(x, dtype=dtype)

def binary_crossentropy(target, output, from_logits=False):
    """Binary crossentropy between an output tensor and a target tensor.
    # Arguments
        target: A tensor with the same shape as `output`.
        output: A tensor.
        from_logits: Whether `output` is expected to be a logits tensor.
            By default, we consider that `output`
            encodes a probability distribution.
    # Returns
        A tensor.
    """
    # Note: tf.nn.sigmoid_cross_entropy_with_logits
    # expects logits, Keras expects probabilities.
    if not from_logits:
        # transform back to logits
        _epsilon = _to_tensor(epsilon(), output.dtype.base_dtype)
        output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
        output = tf.log(output / (1 - output))

    return tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=output)

def mean_entropy(yhat, y):
    return mean(binary_crossentropy(yhat, y))

def smart_entropy(yhat, y):
    dif = (y-yhat)
    false_pos = tf.exp(dif)
    false_neg = tf.exp(-dif)

    bias = mean(yhat)-.5
    is_0_heavy = tf.exp(-bias)
    is_1_heavy = tf.exp(bias)

    weights = (is_0_heavy*false_neg + is_1_heavy*false_pos)/2
    total_of_weights = tf.reduce_sum(weights)
    return (weights*binary_crossentropy(yhat, y))/total_of_weights

    # return false_neg, false_pos, bias, w

import keras
from keras import backend as K

def smart_entropy_loss(yhat, y):
    print(yhat)
    print(y)
    dif = (y-yhat)
    false_pos = K.exp(dif)
    false_neg = K.exp(-dif)

    bias = K.mean(yhat)-.5
    is_0_heavy = K.exp(-bias)
    is_1_heavy = K.exp(bias)

    bce = K.binary_crossentropy(yhat, y)
    print('bce:', bce)
    weights = (is_0_heavy*false_neg + is_1_heavy*false_pos)/2
    print('weights:', weights)
    # weights = K.squeeze(weights, axis=-1)
    # print('weights:', weights)
    # total_of_weights = K.sum(weights)
    # print('total_of_weights:', total_of_weights)
    numerator = weights*bce
    print('numerator:', numerator)
    # divided = numerator/total_of_weights
    # print('divided:', divided)
    result = K.mean( numerator, axis=-1 )
    print('result:', result)
    return result

# Launch the default graph.
with tf.Session() as sess:

    y = tf.constant(sess.run(y_rand))
    yhat = tf.constant(sess.run(yhat_rand))
    print('yhat:')
    print(sess.run(yhat))
    print('y:')
    print(sess.run(y))
    # print('simple crossentropy')
    # print(sess.run(binary_crossentropy(yhat, y)))
    # print('mean of simple x-entropy')
    # print(sess.run(mean_entropy(yhat, y)))

    # fn, fp, bias, w = sess.run(smart_entropy(yhat, y))
    # print("false_neg")
    # print(fn)
    # print("false_pos")
    # print(fp)
    # print(bias)
    # print('weights')
    # print(w)

    loss = sess.run(smart_entropy_loss(yhat, y))
    print('loss:')
    print(loss)
    # loss = sess.run(smart_entropy(1-yhat, 1-y))
    # print('loss:')
    # print(loss)
