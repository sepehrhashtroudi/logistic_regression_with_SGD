import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import accuracy_score as acc



def loadData(directory, name):
    with np.load(directory+name) as data:
        data, target = data["images"], data["labels"]
        pos_class = 2
        neg_class = 9
        data_indx = (target==pos_class) + (target==neg_class)
        data = data[data_indx]/255.0
        target = target[data_indx].reshape(-1, 1)
        target[target==pos_class] = 1
        target[target==neg_class] = 0
        np.random.seed(521)
        rand_indx = np.arange(len(data))
        np.random.shuffle(rand_indx)
        data, target = data[rand_indx], target[rand_indx]
        data = data.reshape(-1, 784)
        target = target.reshape(-1, 1)
        train_data, train_target = data[:3500], target[:3500]
        valid_data, valid_target = data[3500:3600], target[3500:3600]
        test_data, test_target = data[3600:], target[3600:]
    return train_data, train_target, valid_data, valid_target, test_data, test_target

train_data, train_target, valid_data, valid_target, test_data, test_target = loadData("../dataset/","notMNIST.npz")

batch_size = 500
n_epochs = 5000

x = tf.placeholder(dtype=tf.float64, shape=[None, 784], name="x")
y = tf.placeholder(dtype=tf.float64, shape=[None, 1], name="y")

w = tf.Variable(tf.random_normal(shape=[784, 1] , stddev=0.1, dtype=tf.float64), name="weights", dtype=tf.float64)
b = tf.Variable(0.0, dtype=tf.float64)

logit = tf.matmul(x, w) + b
y_predicted = 1.0 / (1.0 + tf.exp(-logit))

rates = [1e-7, 1e-2, 1e-2]
loss1 = -1 * tf.reduce_sum(y * tf.log(y_predicted) + (1 - y) * tf.log(1 - y_predicted))
loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logit))
loss3 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logit)) +\
        tf.constant(0.5 *rates[2], dtype=tf.float64) * tf.pow(tf.linalg.norm(w), 2)
loss=[loss1, loss2, loss3]

optimizer = [tf.train.GradientDescentOptimizer(learning_rate=l).minimize(loss) for loss,l in zip(loss, rates)]
train_loss_list = []
valid_loss_list = []
train_acc_list = []
valid_acc_list = []
loss_indx = 2
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(n_epochs):  # run 100 epochs
        train_loss = 0
        for idx in range(3500//batch_size):
            Input_list = {x: train_data[idx*batch_size:(idx+1)*batch_size],
                          y: train_target[idx*batch_size:(idx+1)*batch_size]}
            _, tl = sess.run([optimizer[loss_indx], loss[loss_indx]], feed_dict=Input_list)
            train_loss += tl
        train_acc_list.append(acc(train_target, np.round(sess.run(y_predicted, feed_dict={x: train_data}))))
        valid_acc_list.append(acc(valid_target, np.round(sess.run(y_predicted, feed_dict={x: valid_data}))))
        train_loss_list.append(train_loss/3500)#number should be used as constant
        valid_loss_list.append(sess.run(loss[loss_indx], feed_dict={x: valid_data, y:valid_target})/100)
    w_value, b_value = sess.run([w, b])
    print(w_value[:10], b_value)


fig, ax = plt.subplots(2, 2)
for a in ax.reshape(-1,1):
    a[0].set_xlabel("epochs")
ax[0][0].plot(train_loss_list[:400], color='red', label='train loss')
ax[0][0].plot(valid_loss_list[:400], color='blue', label='valid loss')
ax[0][0].legend()

ax[0][1].plot(train_loss_list, color='red', label='train loss')
ax[0][1].plot(valid_loss_list, color='blue', label='valid loss')
ax[0][1].legend()

ax[1][0].plot(train_acc_list[:400], color='red', label='train acc')
ax[1][0].plot(valid_acc_list[:400], color='blue', label='valid acc')
ax[1][0].legend()

ax[1][1].plot(train_acc_list, color='red', label='train acc')
ax[1][1].plot(valid_acc_list, color='blue', label='valid acc')
ax[1][1].legend()
plt.show()

