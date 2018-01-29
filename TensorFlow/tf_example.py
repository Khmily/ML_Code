import tensorflow as tf
import numpy as np

x_data = np.random.rand(100).astype(np.float32)
y_data = 0.1 * x_data + 0.3

Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biaes = tf.Variable(tf.zeros([1]))

y = Weights * x_data + biaes

loss = tf.reduce_mean(tf.square(y - y_data))

optimizer = tf.train.GradientDescentOptimizer(0.5)

train = optimizer.minimize(loss)

# initialize_all_variables (from tensorflow.python.ops.variables) is deprecated and will be removed after 2017-03-02.
# init = tf.initialize_all_variables()

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biaes))
