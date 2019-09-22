import tensorflow_probability as tfp
import tensorflow as tf

print(tf.Session().run(tfp.bijectors.Transpose(rightmost_transposed_ndims=2).forward(
    [
        [1, 2, 3, 4],
        [5, 6, 7, 8]
    ])))
