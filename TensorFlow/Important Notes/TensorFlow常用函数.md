1、**tf.argmax(vector, 1)**：返回的是vector中的最大值的索引号，如果vector是一个向量，那就返回一个值；如果是一个矩阵，那就返回一个向量，这个向量的每一个维度都是**相对应矩阵行**的最大值元素的索引号。


```
A = [[1, 5, 2, 3]]
B = [[2, 3, 4], [5, 2, 6], [7, 2, 5], [6, 4, 2]]
with tf.Session() as sess:
    print(sess.run(tf.argmax(A, 1)))
    print(sess.run(tf.argmax(B, 1)))
```

输出：

```
[1]
[2 2 0 0]
```

**tf.argmax(vector, 0)**：返回的是vector中的最大值的索引号，如果vector是一个向量，那就默认第一行向量的下标最大；如果是一个矩阵，那就返回一个向量，这个向量的每一个维度都是**每个矩阵行相同位置的最大值的索引号**。


```
A = [[1, 5, 2, 3]]
B = [[2, 3, 4], [5, 2, 6], [7, 2, 5], [6, 4, 2]]
with tf.Session() as sess:
    print(sess.run(tf.argmax(A, 0)))
    print(sess.run(tf.argmax(B, 0)))

```

输出：


```
[0 0 0 0]
[2 3 1]
```

