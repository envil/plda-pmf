import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt

from tensorflow_probability import edward2 as ed

tfd = tfp.distributions

# tf.enable_eager_execution()

N = 6040  # number of users
M = 3952  # number of movies
D = 10  # Latent factors
stddv_datapoints = 0.5


def probabilistic_matrix_factorization(data_dim, latent_dim, num_datapoints, stddv_datapoints):  # (unmodeled) data
    w = ed.Normal(loc=tf.zeros([data_dim, latent_dim]),
                  scale=2.0 * tf.ones([data_dim, latent_dim]),
                  name="w")  # parameter
    z = ed.Normal(loc=tf.zeros([latent_dim, num_datapoints]),
                  scale=tf.ones([latent_dim, num_datapoints]),
                  name="z")  # parameter
    x = tf.nn.softplus(ed.Normal(loc=tf.matmul(w, z),
                  scale=stddv_datapoints * tf.ones([data_dim, num_datapoints]),
                  name="x"))  # (modeled) data
    return x, (w, z)


log_joint = ed.make_log_joint_fn(probabilistic_matrix_factorization)


def read_data():
    data = np.zeros((N, M))
    f = open("./data/ratings.dat", "r")
    lines = f.readlines()

    for line in lines:
        user, movie, rating, *_ = list(map(int, line.split("::")))
        data[user - 1, movie - 1] = rating

    return data


# def get_indicators(N, M, prob_std=0.8):
#     ind = np.random.binomial(1, prob_std, (N, M))
#     return ind


data = read_data()

# I = get_indicators(N, M)

# train_data = np.where(I == 1, data, np.zeros((N, M)))
# test_data = np.where(I == 0, data, np.zeros((N, M)))
tf.reset_default_graph()


def target(w, z):
    """Unnormalized target density as a function of the parameters."""
    return log_joint(data_dim=N,
                     latent_dim=D,
                     num_datapoints=M,
                     stddv_datapoints=stddv_datapoints,
                     w=w, z=z, x=data)


def variational_model(qw_mean, qw_stddv, qz_mean, qz_stddv):
    qw = ed.Normal(loc=qw_mean, scale=qw_stddv, name="qw")
    qz = ed.Normal(loc=qz_mean, scale=qz_stddv, name="qz")
    return qw, qz


log_q = ed.make_log_joint_fn(variational_model)


def target_q(qw, qz):
    return log_q(qw_mean=qw_mean, qw_stddv=qw_stddv,
                 qz_mean=qz_mean, qz_stddv=qz_stddv,
                 qw=qw, qz=qz)


qw_mean = tf.Variable(np.ones([N, D]), dtype=tf.float32)
qz_mean = tf.Variable(np.ones([D, M]), dtype=tf.float32)
qw_stddv = tf.nn.softplus(tf.Variable(np.ones([N, D]), dtype=tf.float32))
qz_stddv = tf.nn.softplus(tf.Variable(np.ones([D, M]), dtype=tf.float32))

qw, qz = variational_model(qw_mean=qw_mean, qw_stddv=qw_stddv,
                           qz_mean=qz_mean, qz_stddv=qz_stddv)

energy = target(qw, qz)
entropy = -target_q(qw, qz)

elbo = energy + entropy

optimizer = tf.train.AdamOptimizer(learning_rate=0.05)
train = optimizer.minimize(-elbo)

init = tf.global_variables_initializer()

t = []

num_epochs = 500

with tf.Session() as sess:
    sess.run(init)

    for i in range(num_epochs):
        sess.run(train)
        if i % 5 == 0:
            t.append(sess.run([elbo]))

    w_mean_inferred = sess.run(qw_mean)
    w_stddv_inferred = sess.run(qw_stddv)
    z_mean_inferred = sess.run(qz_mean)
    z_stddv_inferred = sess.run(qz_stddv)

print("Inferred axes:")
print(w_mean_inferred)
print("Standard Deviation:")
print(w_stddv_inferred)

plt.plot(range(1, num_epochs, 5), t)
plt.show()

with ed.interception(ed.make_value_setter(w=w_mean_inferred,
                                          z=z_mean_inferred)):
    generate = probabilistic_matrix_factorization(
        data_dim=N, latent_dim=D,
        num_datapoints=M, stddv_datapoints=stddv_datapoints)

with tf.Session() as sess:
    x_generated, _ = sess.run(generate)

plt.scatter(data[:, 0], data[:, 1], color='blue', alpha=0.1, label='Actual data')
plt.scatter(x_generated[0, :], x_generated[1, :], color='red', alpha=0.1, label='Simulated data (VI)')
plt.legend()
plt.axis([-5, 5, -5, 5])
plt.show()

plt.imshow(data, vmin=np.min(data), vmax=np.max(data))
plt.show()
plt.imshow(x_generated, vmin=np.min(x_generated), vmax=np.max(x_generated))
plt.show()
