import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
#读取数据，数据集放在工程目录下
mnist=input_data.read_data_sets('.',one_hot=True)

#定义学习率   placeholder:图运算中的占位符
learning_rate=tf.placeholder(tf.float32)

#x为784维度
x=tf.placeholder(tf.float32,[None,784],name='x')
w=tf.Variable(tf.truncated_normal([784,10]),name='weight')
b=tf.Variable(tf.zeros([10]),name='bais')

#未经激活的输出
logits=tf.matmul(x,w)+b

#标签值
y=tf.placeholder(tf.float32,[None,10],name='y')

#交叉熵损失
cross_entropy=tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=logits)
)

#每次返回的步长
train_step=tf.train.GradientDescentOptimizer(learning_rate).minize(cross_entropy)

correct_prediction=tf.equal(tf.arg_max(y,1),tf.arg_max(logits,1))

#tf.cast---将布尔型转换为float型
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#查看节点node
grap=tf.get_default_graph()
print(grap)
print(grap.as_graph_def())

#开启一个计算的session
sess=tf.Session()

#节点的初始化
sess.run(tf.global_variables_initializer())

#分批次操作
lr=1.0
for step in range(3000):
    if step>1000:
        lr=0.3
    if step>2000:
        lr=0.1
    batch_x,batch_y=mnist.train.next_batch(32)
    _,loss=sess.run([train_step,cross_entropy],
                    feed_dict={
                        x:batch_x,
                        y:batch_y,
                        learning_rate:lr
                    })
    #实时查看进度
    if(step+1)%100 ==0:
        print('#'*10)
        print('step[{}],entropy loss:[{}]'.format(step+1,loss))
        #训练集上的准确率
        print(sess.run(accuracy,feed_dict={x:batch_x,y:batch_y}))

        #测试集上的准确率
        print(sess.run(
            accuracy,
            feed_dict={
                x:mnist.test.images,
                y:mnist.test.labels
            }
        ))





