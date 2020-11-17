import glob
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator


studentID_name = "2016113290_김민규"  # 자신의 학번과 이름으로 수정
dropout_rate = 0.5  # 드롭아웃 비율 설정
train_epoch = 1001


def build_CNN_classifier(x):
    W_conv1 = tf.Variable(tf.truncated_normal(shape=[5, 5, 3, 64], stddev=5e-2))
    b_conv1 = tf.Variable(tf.constant(0.1, shape=[64]))
    h_conv1 = tf.nn.selu(tf.nn.conv2d(x, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)

    # 첫번째 Pooling layer
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 두번째 convolutional layer - 64개의 특징들(feature)을 64개의 특징들(feature)로 맵핑(maping)합니다.
    W_conv2 = tf.Variable(tf.truncated_normal(shape=[5, 5, 64, 64], stddev=5e-2))
    b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))
    h_conv2 = tf.nn.selu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)

    # 두번째 pooling layer.
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 세번째 convolutional layer
    W_conv3 = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 128], stddev=5e-2))
    b_conv3 = tf.Variable(tf.constant(0.1, shape=[128]))
    h_conv3 = tf.nn.sigmoid(tf.nn.conv2d(h_pool2, W_conv3, strides=[1, 1, 1, 1], padding='SAME') + b_conv3)

    # Fully Connected Layer 1 - 2번의 downsampling 이후에, 우리의 32x32 이미지는 8x8x128 특징맵(feature map)이 됩니다.
    # 이를 384개의 특징들로 맵핑(maping)합니다.
    W_fc1 = tf.Variable(tf.truncated_normal(shape=[8 * 8 * 128, 384], stddev=5e-2))
    b_fc1 = tf.Variable(tf.constant(0.1, shape=[384]))

    h_conv3_flat = tf.reshape(h_conv3, [-1, 8 * 8 * 128])
    h_fc1 = tf.nn.sigmoid(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)
    h_fc1 = tf.layers.batch_normalization(h_fc1)

    # Dropout - 모델의 복잡도를 컨트롤합니다. 특징들의 co-adaptation을 방지합니다.
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Fully Connected Layer 2 - 384개의 특징들(feature)을 10개의 클래스-airplane, automobile, bird...-로 맵핑(maping)합니다.
    W_fc2 = tf.Variable(tf.truncated_normal(shape=[384, 2], stddev=5e-2))
    b_fc2 = tf.Variable(tf.constant(0.1, shape=[2]))
    logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    logits = tf.layers.batch_normalization(logits)

    return logits


def load_image(dir):
    folders = glob.glob(dir + "*")

    label = []
    train_x_data = []
    train_y_label = []

    for i in range(len(folders)):
        label.append(folders[i].split("\\")[1])
        image_dir = glob.glob(dir + label[i] + "/*.jpg")

        for j in range(len(image_dir)):
            train_x_data.append(plt.imread(image_dir[j]))

            if i == 0:
                train_y_label.append((1, 0))
            else:
                train_y_label.append((0, 1))

    train_x_data = np.array(train_x_data, dtype=np.int32).reshape(-1, 32, 32, 3)
    train_y_label = np.array(train_y_label, dtype=np.int32).reshape(-1, 2)
    label = np.array(label)

    idx = np.arange(train_x_data.shape[0])
    np.random.shuffle(idx)
    train_x_data = train_x_data[idx]
    train_y_label = train_y_label[idx]

    return train_x_data, train_y_label, label


dir = "./dataset/"
train_x_data, train_y_label, label = load_image(dir)

validation_x_data = train_x_data[1000:1600]
validation_y_label = train_y_label[1000:1600]
train_x_data = train_x_data[:1000]
train_y_label = train_y_label[:1000]

tf.reset_default_graph()

x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name="x")
y = tf.placeholder(tf.float32, shape=[None, 2], name="y")
keep_prob = tf.placeholder(tf.float32, name="keep_prob")

logits = build_CNN_classifier(x)
y_pred = tf.nn.softmax(logits, name="y_pred")

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
train_step = tf.train.RMSPropOptimizer(1e-3).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

save_dir = studentID_name
saver = tf.train.Saver()
checkpoint_path = os.path.join(studentID_name, "model")
ckpt = tf.train.get_checkpoint_state(save_dir)

sess = tf.Session()
# 모든 변수들을 초기화한다.
sess.run(tf.global_variables_initializer())

# 10000 Step만큼 최적화를 수행합니다.
for i in range(train_epoch):

    # 100 Step마다 training 데이터셋에 대한 정확도와 loss를 출력합니다.
    if i % 100 == 0:
        train_accuracy = accuracy.eval(session=sess, feed_dict={x: train_x_data, y: train_y_label, keep_prob: 1.0})
        loss_print = loss.eval(session=sess, feed_dict={x: train_x_data, y: train_y_label, keep_prob: 1.0})
        print("반복(Epoch): %d, 트레이닝 데이터 정확도: %f, 손실 함수(loss): %f" % (i, train_accuracy, loss_print))
        saver.save(sess, checkpoint_path, global_step=i)
    # 20% 확률의 Dropout을 이용해서 학습을 진행합니다.
    sess.run(train_step, feed_dict={x: train_x_data, y: train_y_label, keep_prob: dropout_rate})

test_accuracy = accuracy.eval(session=sess, feed_dict={x: validation_x_data, y: validation_y_label, keep_prob: 1.0})
loss_print = loss.eval(session=sess, feed_dict={x: validation_x_data, y: validation_y_label, keep_prob: 1.0})
print("검증 데이터 정확도: %f, 손실 함수(loss): %f" % (test_accuracy, loss_print))

sess.close()

