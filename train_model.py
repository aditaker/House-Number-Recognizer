import read
import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf



def cnn_model( features, labels, mode):


	input_layer = tf.reshape(features['x'],[-1, 32, 32, 3])
	print("yopooo")
	conv1 = tf.layers.conv2d(
	  inputs=input_layer,
	  filters=32,
	  kernel_size=[3, 3],
	  padding="same",
	  activation=tf.nn.relu)
	print("hyyyyyy")

	pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

	conv2 = tf.layers.conv2d(
	  inputs=pool1,
	  filters=48,
	  kernel_size=[3, 3],
	  padding="same",
	  activation=tf.nn.relu)

	pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

	conv3 = tf.layers.conv2d(
	  inputs=pool2,
	  filters=64,
	  kernel_size=[3, 3],
	  padding="same",
	  activation=tf.nn.relu)

	pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

	conv4 = tf.layers.conv2d(
	  inputs=pool3,
	  filters=128,
	  kernel_size=[3, 3],
	  padding="same",
	  activation=tf.nn.relu)

	pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

	# conv5 = tf.layers.conv2d(
	#   inputs=pool4,
	#   filters=192,
	#   kernel_size=[3, 3],
	#   padding="same",
	#   activation=tf.nn.relu)

	# pool5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[2, 2], strides=2)

	# Dense Layer
	pool4_flat = tf.reshape(pool4, [-1, 2 * 2 * 128])
	
	dense1 = tf.layers.dense(inputs=pool4_flat, units=1024, activation=tf.nn.relu)
	dropout1 = tf.layers.dropout(
	  inputs=dense1, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
	logit1 = tf.layers.dense(inputs=dropout1, units=11)
	#logit1 = tf.nn.softmax(fc1, name="softmax_tensor")

	dense2 = tf.layers.dense(inputs=pool4_flat, units=1024, activation=tf.nn.relu)
	dropout2 = tf.layers.dropout(
	  inputs=dense2, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
	logit2 = tf.layers.dense(inputs=dropout2, units=11)
	#logit2 = tf.nn.softmax(fc2, name="softmax_tensor")

	dense3 = tf.layers.dense(inputs=pool4_flat, units=1024, activation=tf.nn.relu)
	dropout3 = tf.layers.dropout(
	  inputs=dense3, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
	logit3 = tf.layers.dense(inputs=dropout3, units=11)
	#logit3 = tf.nn.softmax(fc3, name="softmax_tensor")

	dense4 = tf.layers.dense(inputs=pool4_flat, units=1024, activation=tf.nn.relu)
	dropout4 = tf.layers.dropout(
	  inputs=dense4, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
	logit4 = tf.layers.dense(inputs=dropout4, units=11)
	#logit4 = tf.nn.softmax(fc4, name="softmax_tensor")

	dense5 = tf.layers.dense(inputs=pool4_flat, units=1024, activation=tf.nn.relu)
	dropout5 = tf.layers.dropout(
	  inputs=dense5, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
	logit5 = tf.layers.dense(inputs=dropout5, units=11)
	#logit5 = tf.nn.softmax(fc5, name="softmax_tensor")

	logits = tf.stack([logit1, logit2, logit3, logit4, logit5])
	pred_cls = tf.transpose(tf.argmax(logits, axis=2))

	# predictions = {
	#   # Generate predictions (for PREDICT and EVAL mode)
	#   "classes": tf.argmax(input=logits, axis=1),
	#   # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
	#   # `logging_hook`.
	#   "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
	# }

	# if mode == tf.estimator.ModeKeys.PREDICT:
	# 	return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

	# Calculate Loss (for both TRAIN and EVAL modes)

	loss1 = tf.losses.sparse_softmax_cross_entropy(labels=labels[:,0], logits=logit1)
	loss2 = tf.losses.sparse_softmax_cross_entropy(labels=labels[:,1], logits=logit2)
	loss3 = tf.losses.sparse_softmax_cross_entropy(labels=labels[:,2], logits=logit3)
	loss4 = tf.losses.sparse_softmax_cross_entropy(labels=labels[:,3], logits=logit4)
	loss5 = tf.losses.sparse_softmax_cross_entropy(labels=labels[:,4], logits=logit5)

	loss = loss1+loss2+loss3+loss4+loss5;
	# Configure the Training Op (for TRAIN mode)
	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
		train_op = optimizer.minimize(
			loss=loss,
			global_step=tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

	# Add evaluation metrics (for EVAL mode)
	eval_metric_ops = {
	  "accuracy": tf.metrics.accuracy(
		  labels=labels, predictions=pred_cls)}
	return tf.estimator.EstimatorSpec(
	  mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)




svhn_classifier = tf.estimator.Estimator( model_fn=cnn_model, model_dir="svhn_convnet_model")

tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(
  tensors=tensors_to_log, every_n_iter=50)

train_inp, train_out = read.pre_process('train')
train_inp, train_out = shuffle(train_inp, train_out)
# train_out = np.array([[1,2,3,4,5]],dtype=int)
# train_inp = np.zeros((1,32,32,3), dtype=np.float16)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
	x={"x": train_inp},
	y=train_out,
	batch_size=100,
	num_epochs=None,
	shuffle=True)
svhn_classifier.train(
	input_fn=train_input_fn,
	steps=100)



