# import keras
import tensorflow as tf


def binary_crossentropy(target, output, from_logits=False):
    if not from_logits:
        # transform back to logits
        _epsilon = tf.convert_to_tensor(1e-7, dtype=output.dtype.base_dtype)
        output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
        output = tf.log(output / (1 - output))

    return tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=output)


def focal(alpha=0.25, gamma=2.0):
    def _focal(y_true, y_pred):
        labels = y_true
        classification = y_pred

        # compute the focal loss
        alpha_factor = tf.ones_like(labels) * alpha
        alpha_factor = tf.where(tf.equal(labels, 1), alpha_factor, 1 - alpha_factor)
        focal_weight = tf.where(tf.equal(labels, 1), 1 - classification, classification)
        focal_weight = alpha_factor * focal_weight ** gamma

        cls_loss = focal_weight * binary_crossentropy(labels, classification)

        # filter out "ignore" anchors
        anchor_state = tf.reduce_max(labels, axis=2, keepdims=False)  # -1 for ignore, 0 for background, 1 for object
        indices = tf.where(tf.not_equal(anchor_state, -1))
        cls_loss = tf.gather_nd(cls_loss, indices)

        # compute the normalizer: the number of positive anchors
        normalizer = tf.where(tf.equal(anchor_state, 1))
        normalizer = tf.cast(tf.shape(normalizer)[0], 'float32')
        normalizer = tf.maximum(1.0, normalizer)

        return tf.reduce_sum(cls_loss, keepdims=False) / normalizer

    return _focal


def smooth_l1(sigma=3.0):
    sigma_squared = sigma ** 2

    def _smooth_l1(y_true, y_pred):
        # separate target and state
        regression = y_pred
        regression_target = y_true[:, :, :4]
        anchor_state = y_true[:, :, 4]

        # compute smooth L1 loss
        # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
        #        |x| - 0.5 / sigma / sigma    otherwise
        regression_diff = regression - regression_target
        regression_diff = tf.abs(regression_diff)
        regression_loss = tf.where(tf.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * tf.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )

        # filter out "ignore" anchors
        indices = tf.where(tf.equal(anchor_state, 1))
        regression_loss = tf.gather_nd(regression_loss, indices)

        # compute the normalizer: the number of positive anchors
        normalizer = tf.maximum(1, tf.shape(indices)[0])
        normalizer = tf.cast(tf.maximum(1, normalizer), dtype='float32')

        return tf.reduce_sum(regression_loss, keepdims=False) / normalizer

    return _smooth_l1
