import pickle
import os
import numpy as np
import tensorflow as tf


# ===================================================================================== #

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict


def load_data():
    train_batches = {}
    test_batch = []

    # Load Test and Train Batches
    for i in range(6):

        if i == 5:
            data_dict = unpickle('./cifar-10-batches-py/test_batch')
        else:
            data_dict = unpickle('./cifar-10-batches-py/data_batch_{}'.format(i + 1))

        X = np.array([im.transpose([1, 2, 0]) for im in np.reshape(data_dict['data'],
                                                                   newshape=(10000, 3, 32, 32))]) / 255.0
        y = np.array(data_dict['labels'])
        y = np.eye(10)[y]

        if i == 5:
            test_batch = [X, y]
        else:
            train_batches[i] = [X, y]

    return train_batches, test_batch


# ===================================================================================== #


def main():
    # ================================================================================ #
    # Hyper parameters.
    # ================================================================================ #
    num_epochs = 500
    batches_num = 5

    # ================================================================================ #
    # Read Data.
    # ================================================================================ #
    train_batches, test_batch = load_data()

    # ================================================================================ #
    # Build CNN and Define optimizer.
    # ================================================================================ #
    net_input = tf.placeholder(tf.float32, shape=(None, 32, 32, 3), name="input")
    labels_encoded = tf.placeholder(tf.float64, shape=(None, 10), name="labels_encoded")

    conv_1 = tf.layers.conv2d(inputs=net_input, filters=16, padding="SAME", kernel_size=[5, 5], activation=tf.nn.relu)
    pool_1 = tf.nn.max_pool(value=conv_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    conv_2 = tf.layers.conv2d(inputs=pool_1, filters=20, padding="SAME", kernel_size=[5, 5], activation=tf.nn.relu)
    pool_2 = tf.nn.max_pool(value=conv_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    conv_3 = tf.layers.conv2d(inputs=pool_2, filters=20, padding="SAME", kernel_size=[5, 5], activation=tf.nn.relu)
    pool_3 = tf.nn.max_pool(value=conv_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    pool_3_flatten = tf.reshape(pool_3, [-1, 4 * 4 * 20])

    output = tf.layers.dense(inputs=pool_3_flatten, units=10, activation=tf.identity)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_encoded, logits=output))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    # ================================================================================ #
    # Test tensors.
    # ================================================================================ #
    predictions = tf.argmax(tf.nn.softmax(output), axis=1)
    labels = tf.argmax(labels_encoded, axis=1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(labels, predictions), tf.float64))

    # ================================================================================ #
    # Initialize Tensorflow objects.
    # ================================================================================ #
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    # ================================================================================ #
    # Create and run training session.
    # ================================================================================ #
    with tf.Session() as training_sess:

        training_sess.run(init)

        for epoch in range(num_epochs):

            for batch_num in range(batches_num):
                xs = train_batches[batch_num][0]
                ys = train_batches[batch_num][1]

                _, c, acc = training_sess.run([optimizer, cost, accuracy],
                                              feed_dict={net_input: xs, labels_encoded: ys})

                print 'Epoch:{}, Batch:{}, Cost:{}, Accuracy:{}'.format(epoch, batch_num, c, acc)

        saver.save(training_sess, os.path.join('./model'), global_step=num_epochs)

        print "Optimization Finished!\n"

    # ================================================================================ #
    # Create and run training session.
    # ================================================================================ #
    with tf.Session() as test_sess:

        saver.restore(test_sess, tf.train.latest_checkpoint('./'))

        xs = test_batch[0]
        ys = test_batch[1]

        acc, c_test = test_sess.run([accuracy, cost], feed_dict={net_input: xs, labels_encoded: ys})

        print "Test Accuracy:{}".format(acc)
        print "Test Cost:{}".format(c_test)


if __name__ == '__main__':
    main()
