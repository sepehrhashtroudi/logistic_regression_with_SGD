import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


n_epoch = 500
batch_size = 500
loss_list =[]
test_accuracy_list =[]


def load_Data():
    with np.load("notMNIST.npz") as data :
        Data, Target = data ["images"], data["labels"]
        posClass = 2
        negClass = 9
        dataIndx = (Target==posClass) + (Target==negClass)
        Data = Data[dataIndx]/255.
        Target = Target[dataIndx].reshape(-1, 1)
        Target[Target==posClass] = 1
        Target[Target==negClass] = 0
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]
        trainData, trainTarget = Data[:3500].reshape([-1,784]), Target[:3500]
        validData, validTarget = Data[3500:3600].reshape([-1,784]), Target[3500:3600]
        testData, testTarget = Data[3600:].reshape([-1,784]), Target[3600:]
        return trainData, trainTarget, validData, validTarget, testData, testTarget


def plot_digit(some_digit):
    some_digit_image = some_digit.reshape(28, 28)
    plt.imshow(some_digit_image, cmap=plt.cm.binary, interpolation="nearest")
    plt.axis("off")
    plt.show()

def plot_loss_acc(loss_acc_list):
    plt.plot(loss_acc_list)
    plt.show()

def accuracy(test_data, test_target, w , b):
    error = 0
    for i in range(len(test_target)):
        predicted_target = np.round(1.0 / (1 + np.exp(np.matmul(test_data[i], w) + b)))
        if(predicted_target != test_target[i] ):
            error += 1
    return 100*error/len(test_target)

def log_reg():
    X = tf.placeholder(dtype=tf.float64, shape=[None, 784], name="x")
    Y = tf.placeholder(dtype=tf.float64, shape=[None,1], name="y")
    w = tf.Variable(tf.random_normal(shape=[784, 1], stddev=0.01, dtype=tf.float64), name="weights", dtype=tf.float64)
    b = tf.Variable(0.0, dtype=tf.float64)
    Y_predicted = 1.0 / (1 + tf.exp( tf.matmul(X , w) + b))
    # print("a", x_train[1])
    loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=Y_predicted))

    # eps = 1e-10
    # maximum_likelihood = Y*tf.log(Y_predicted+eps) + (1-Y)*tf.log(1-Y_predicted+eps)
    # loss = -1 * tf.reduce_sum(maximum_likelihood)


    # loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=Y_predicted))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001).minimize(loss)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(n_epoch):  # run 100 epochs
            for idx in range(int(len(x_train)/batch_size)):
                Input_list = {X: x_train[idx*batch_size:(idx+1)*batch_size], Y: y_train[idx*batch_size:(idx+1)*batch_size]}
                _,Loss,w_value,b_value = sess.run([optimizer,loss,w,b], feed_dict=Input_list)
                loss_list. append(Loss)
            test_accuracy_list.append(accuracy(x_test, y_test, w_value, b_value))


        # print(w_value,b_value)
        return w_value , b_value





x_train, y_train, x_valid, y_valid, x_test, y_test = load_Data()
# plot_digit(x_test[1])
w,b =log_reg()
plot_loss_acc(loss_list)
plot_loss_acc(test_accuracy_list)
