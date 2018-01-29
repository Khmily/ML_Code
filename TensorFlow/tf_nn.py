import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def add_layer(inputs, input_size, output_size, activation_function=None):
    """
    :param inputs: 输入的数据
    :param input_size: 输入数据的维数
    :param output_size: 输入数据的维数
    :param activation_function: 激活函数，默认为None
    :return: 返回输出值
    """
    Weights = tf.Variable(tf.random_normal([input_size, output_size]))
    biaes = tf.Variable(tf.zeros([1, output_size]) + 0.1)
    output_hat = tf.matmul(inputs, Weights) + biaes
    if activation_function is None:
        output = output_hat
    else:
        output = activation_function(output_hat)
    return output


x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
# 模拟数据噪音
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

# 经过隐藏层后的输出值
layer1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# 经过输出层后的输出值，即为最后的预测值
layer2 = add_layer(layer1, 10, 1, activation_function=None)

loss = tf.reduce_mean((tf.reduce_sum(tf.square(ys - layer2), reduction_indices=[1])))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

# 结果可视化
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_data, y_data)
plt.ion()
plt.show()

for i in range(1000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        # print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
        # 结果可视化
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value = sess.run(layer2, feed_dict={xs: x_data})
        lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
        plt.pause(0.1)
