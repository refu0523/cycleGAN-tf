import tensorflow as tf

def leaky_relu(x, leak=0.2, name="leaky_relu"):
	     with tf.variable_scope(name):
	         f1 = 0.5 * (1 + leak)
	         f2 = 0.5 * (1 - leak)
	         return f1 * x + f2 * abs(x)

def instance_norm(x, name="instance_norm"):
	with tf.variable_scope(name):
	    epsilon = 1e-5
	    mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
	    scale = tf.get_variable('scale',[x.get_shape()[-1]], 
	        initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.02))
	    offset = tf.get_variable('offset',[x.get_shape()[-1]],initializer=tf.constant_initializer(0.0))
	    out = scale*tf.div(x-mean, tf.sqrt(var+epsilon)) + offset
	    return out

def conv2d(x, num_outputs, kernel_size=4, strides=2, padding = "SAME", stddev = 0.02, name="conv_2d"):
	return tf.contrib.layers.conv2d(x, num_outputs, kernel_size, strides, padding,\
	  activation_fn=None, weights_initializer=tf.truncated_normal_initializer(stddev=stddev),biases_initializer=tf.constant_initializer(0.0))
	
def deconv2d(x, num_outputs, kernel_size=4, strides=2, padding = "SAME", stddev = 0.02, name="conv_2d"):
	return tf.contrib.layers.conv2d_transpose(x, num_outputs, kernel_size, strides, padding,\
	  activation_fn=None, weights_initializer=tf.truncated_normal_initializer(stddev=stddev),biases_initializer=tf.constant_initializer(0.0))

def residule_block(x, dim, name="residule_block"):
	with tf.variable_scope(name) as scope:
		y = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
		y = conv2d(y, dim, 3, strides=1, padding='VALID', name='r_conv_1')
		y = tf.nn.relu(instance_norm(y,'r_bn_1'))

		y = tf.pad(y, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
		y = conv2d(y, dim, 3, strides=1, padding='VALID', name='r_conv_2')
		y = instance_norm(y,'r_bn_2')
		return y+x

