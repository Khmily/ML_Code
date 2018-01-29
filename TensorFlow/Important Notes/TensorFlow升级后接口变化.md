1、
`tf.train.SummaryWriter("logs/", sess.graph)
`不推荐使用，在2016-11-30之后删除。 

更新说明：

使用`tf.summary.FileWriter("logs/", sess.graph)`

另外再启动tensorboard时，` tensorboard --logdir='logs/'`不再使用，**logs的目录并不需要加引号**, logs 中有多个event 时，会生成scalar 的对比图，但 graph 只会展示最新的结果，应该使用`tensorboard --logdir=logs`。你会发现此时运行后还会报如下错误：
```
tensorboard：No graph definition files were found
```

解决方法:

原因：选项“–logdir”的值，应该是绝对路径。

正确方法：

```
 tensorboard --logdir=E:\ML_Code\tf_test\logs
 ```

---

2、summary独立出来了，以前`tf.XXX_summary`这样的下划线变成了`tf.summary.XXX`的格式。

- 对于标量

如果我们想对标量在训练中可视化，可以使用`tf.summary.scalar()`，比如损失loss：


```
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1])) 
tf.summary.scalar('loss',loss)
```
得到一个loss的summary。

- 对于参数

应使用`tf.summary.histogram()`，如全链接的权重：


```
tf.summary.histogram("/weights",Weights)
```
 

- merge并运行

就像变量需要初始化一样，summary也需要merge：

```
merged = tf.summary.merge_all()
```


之后定义一个输出器记录下在运行中的数据：


```
writer = tf.summary.FileWriter("output/",sess.graph)
```


最后记得在训练过程中执行这两个模块：


```
for i in range(1000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i%50==0:# 50次记录一次
        result = sess.run(merged,feed_dict={xs:x_data,ys:y_data})
        writer.add_summary(result,i)
```


---

3、`tf.initialize_all_variables()`不推荐使用，在2017-03-02之后删除。

更新说明：

使用`tf.global_variables_initializer()`

