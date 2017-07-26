# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 14:31:19 2017

@author: celes
"""
from __future__ import print_function
from __future__ import division
from glob import glob
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import time
from util import *
from ops import *
import matplotlib.gridspec as gridspec
import os

class cycleGAN():
	def __init__(self, batch_size = 1, epoch = 200, height = 256, width = 256, in_depth = 3,\
	 out_depth = 3, ngf = 32, ndf = 64, lam =10, stddev =0.02, dataset = 'horse2zebra', pool_size = 50, train_size=1e8):
		self.batch_size = batch_size
		self.epoch = epoch
		self.height = height
		self.width = height
		self.in_depth = in_depth
		self.out_depth = out_depth
		self.ndf = ndf
		self.ngf = ngf
		self.lam = lam
		self.dataset = dataset
		self.stddev = stddev
		self.pool = ImagePool(pool_size)
		self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
		self.train_size = train_size

	def discriminator(self, x, name, reuse = False):
		with tf.variable_scope(name) as scope:
			if reuse:
				scope.reuse_variables()
			#CNN
			x = conv2d(x, self.ndf, 5, strides=2, padding='SAME', name='d_conv_1')
			x = leaky_relu(instance_norm(x,'d_bn_1'))
			x = conv2d(x, self.ndf*2, 5, strides=2, padding='SAME', name='d_conv_2')
			x = leaky_relu(instance_norm(x,'d_bn_2'))
			x = conv2d(x, self.ndf*4, 5, strides=2, padding='SAME', name='d_conv_3')
			x = leaky_relu(instance_norm(x,'d_bn_3'))
			x = conv2d(x, self.ndf*8, 5, strides=1, padding='SAME', name='d_conv_4')
			x = leaky_relu(instance_norm(x,'d_bn_4'))
			out = conv2d(x, 1, 5, strides=1, padding='SAME', name = "d_pred_1")
			return out

	def generator(self, x, name, reuse = False):
		with tf.variable_scope(name) as scope:
			if reuse:
				scope.reuse_variables()

			#encode  ngf=32
			x = tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
			x = conv2d(x, self.ngf, 7, strides=1, padding='VALID', name='g_conv_1')
			x = tf.nn.relu(instance_norm(x,'g_bn_1'))
			x = conv2d(x, self.ngf*2, 3, strides=2, padding='SAME', name='g_conv_2')
			x = tf.nn.relu(instance_norm(x,'g_bn_2'))
			x = conv2d(x, self.ngf*4, 3, strides=2, padding='SAME', name='g_conv_3')
			x = tf.nn.relu(instance_norm(x,'g_bn_3'))

			
			#transform-resnet 3 block
			x = residule_block(x, self.ngf*4, name='g_r_1')
			x = residule_block(x, self.ngf*4, name='g_r_2')
			x = residule_block(x, self.ngf*4, name='g_r_3')
			x = residule_block(x, self.ngf*4, name='g_r_4')
			x = residule_block(x, self.ngf*4, name='g_r_5')
			x = residule_block(x, self.ngf*4, name='g_r_6')
			#decode
			x = deconv2d(x, self.ngf*2, 3, strides=2, padding='SAME', name='g_dconv_1')
			x = tf.nn.relu(instance_norm(x, name='g_d1_bn'))
			x = deconv2d(x, self.ngf, 3, strides=2, padding='SAME', name='g_dconv_2')
			x = tf.nn.relu(instance_norm(x, name='g_d2_bn'))
			x = tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")

			x = conv2d(x, self.out_depth, 7, strides=1, padding='VALID', name='g_pred_1')
			out = tf.nn.tanh(instance_norm(x, name='g_pred_bn'))
			return out

	def train(self):
		saver = tf.train.Saver() # defaults to saving all variables - in this case embed_matrix, nce_weight, nce_bias
		data = tf.placeholder(tf.float32, [None, self.height, self.width, self.in_depth+self.out_depth], name ="d_real_AB")
		fake_sample_A = tf.placeholder(tf.float32, [None, self.height, self.width, self.in_depth], name='fake_A_sample')
		fake_sample_B = tf.placeholder(tf.float32,[None, self.height, self.width, self.out_depth],  name='fake_B_sample')

		input_A = data[:, :, :, :self.in_depth]
		input_B = data[:, :, :, self.in_depth:self.in_depth + self.out_depth]
		# transfer A2B, B2A
		fake_B = self.generator(input_A, name="g_A2B")
		fake_A = self.generator(input_B, name="g_B2A")
		# reproduce A,B
		gen_fake_A = self.generator(fake_B, name="g_A2B", reuse=True)
		gen_fake_B = self.generator(fake_A, name="g_B2A", reuse=True)
		
		# discriminate fake_B and A, fake_A and B
		d_fake_B = self.discriminator(fake_B, name="d_B")
		d_fake_A = self.discriminator(fake_A, name="d_A")

		# input should be close to reproduction
		cyc_loss = tf.reduce_mean(tf.abs(input_A - gen_fake_A)) + tf.reduce_mean(tf.abs(input_B - gen_fake_B))
		
		# generator hope transfered A,B should close to input (1)
		g_loss_A2B = tf.reduce_mean(tf.squared_difference(d_fake_B,1)) + self.lam * cyc_loss
		g_loss_B2A = tf.reduce_mean(tf.squared_difference(d_fake_A,1)) + self.lam * cyc_loss


		d_input_A = self.discriminator(input_A, name="d_A", reuse=True)
		d_input_B = self.discriminator(input_B, name="d_B", reuse=True)
		
		# pool
		d_fake_sample_A = self.discriminator(fake_sample_A, name="d_A", reuse=True)
		d_fake_sample_B = self.discriminator(fake_sample_B, name="d_B", reuse=True)

		d_loss_real_A = tf.reduce_mean(tf.squared_difference(d_input_A,1))
		d_loss_fake_A = tf.reduce_mean(tf.squared_difference(d_fake_sample_A,0))
		d_loss_A = (d_loss_real_A + d_loss_fake_A) / 2

		d_loss_real_B = tf.reduce_mean(tf.squared_difference(d_input_B,1))
		d_loss_fake_B = tf.reduce_mean(tf.squared_difference(d_fake_sample_B,0))
		d_loss_B = (d_loss_real_B + d_loss_fake_B) / 2

		g_loss_A2B_sum = tf.summary.scalar("g_loss_A2B", g_loss_A2B)
		g_loss_B2A_sum = tf.summary.scalar("g_loss_B2A", g_loss_B2A)
		d_loss_B_sum = tf.summary.scalar("d_loss_B", d_loss_B)
		d_loss_A_sum = tf.summary.scalar("d_loss_A", d_loss_A)
		
		d_loss_real_A_sum = tf.summary.scalar("d_loss_real_A", d_loss_real_A)
		d_loss_real_B_sum = tf.summary.scalar("d_loss_real_B", d_loss_real_B)
		d_loss_fake_A_sum = tf.summary.scalar("d_loss_fake_A", d_loss_fake_A)
		d_loss_fake_B_sum = tf.summary.scalar("d_loss_fake_B", d_loss_fake_B)

		d_loss_B_sum = tf.summary.merge([d_loss_B_sum, d_loss_real_B_sum, d_loss_fake_B_sum])
		d_loss_A_sum = tf.summary.merge([d_loss_A_sum, d_loss_real_A_sum, d_loss_fake_A_sum])
		
		# test
		#test_A = tf.placeholder(tf.float32, [None, self.height, self.width, self.in_depth], name='test_A')
		#test_B = tf.placeholder(tf.float32,[None, self.height, self.width, self.out_depth],  name='test_B')

		#test_out_B = self.generator(test_A, name="g_A2B", reuse=True)
		#test_out_A = self.generator(test_B, name="g_B2A", reuse=True)


		d_solver_A = tf.train.AdamOptimizer(0.0002, 0.5).minimize(d_loss_A, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='d_A'))
		d_solver_B = tf.train.AdamOptimizer(0.0002, 0.5).minimize(d_loss_B, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='d_B'))
		g_solver_A2B = tf.train.AdamOptimizer(0.0002, 0.5).minimize(g_loss_A2B, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='g_A2B'))
		g_solver_B2A = tf.train.AdamOptimizer(0.0002, 0.5).minimize(g_loss_B2A, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='g_B2A'))

		
		with tf.Session() as sess:
			initial_step = 0
			counter = 0
			start_time = time.time()

			if not os.path.exists('out/'):
				os.makedirs('out/')
			if not os.path.exists('checkpoints/'):
				os.makedirs('checkpoints/')

			sess.run(tf.global_variables_initializer())
			writer = tf.summary.FileWriter("./logs", sess.graph)

			ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))
			if ckpt and ckpt.model_checkpoint_path:
				saver.restore(sess, ckpt.model_checkpoint_path)
				initial_step  = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])


			for epoch in range(initial_step,self.epoch):
				data_A = glob('./datasets/{}/*.*'.format(self.dataset + '/trainA'))
				data_B = glob('./datasets/{}/*.*'.format(self.dataset + '/trainB'))
				np.random.shuffle(data_A)
				np.random.shuffle(data_B)
				batch_idxs = min(min(len(data_A), len(data_B)), self.train_size) // self.batch_size
				for i in range(0, batch_idxs):
					batch_files = list(zip(data_A[i * self.batch_size:(i + 1) * self.batch_size],
						data_B[i * self.batch_size:(i + 1) * self.batch_size]))
					batch_images = [load_data(batch_file) for batch_file in batch_files]
					batch_images = np.array(batch_images).astype(np.float32)

					fake_pool_A, fake_pool_B = self.pool(sess.run([fake_A, fake_B], feed_dict={data: batch_images}))
					_, summary_str =sess.run([g_solver_A2B, g_loss_A2B_sum], feed_dict ={data: batch_images})
					writer.add_summary(summary_str, counter)

					_, summary_str =sess.run([d_solver_B, d_loss_B_sum], feed_dict ={data: batch_images, fake_sample_B: fake_pool_B})
					writer.add_summary(summary_str, counter)

					_, summary_str =sess.run([g_solver_B2A, g_loss_B2A_sum], feed_dict ={data: batch_images})
					writer.add_summary(summary_str, counter)

					_, summary_str =sess.run([d_solver_A, d_loss_A_sum], feed_dict ={data: batch_images, fake_sample_A: fake_pool_A})
					writer.add_summary(summary_str, counter)

					if i % 50 == 0:
						data_test_A = glob('./datasets/{}/*.*'.format(self.dataset + '/testA'))
						data_test_B = glob('./datasets/{}/*.*'.format(self.dataset + '/testB'))
						np.random.shuffle(data_test_A)
						np.random.shuffle(data_test_B)
						batch_files = list(zip(data_test_A[:self.batch_size], data_test_B[:self.batch_size]))
						sample_images = [load_data(batch_file, False, True) for batch_file in batch_files]
						sample_images = np.array(sample_images).astype(np.float32)

						transfer_B2A, transfer_A2B = sess.run([fake_A, fake_B], feed_dict={data: sample_images})
						save_images(transfer_B2A, [self.batch_size, 1],'./{}/A_{:02d}_{:04d}.jpg'.format('out', epoch, i))
						save_images(transfer_A2B, [self.batch_size, 1],'./{}/B_{:02d}_{:04d}.jpg'.format('out', epoch, i))
					
						saver.save(sess, 'checkpoints/cycleGAN', counter)
					
					print(("Epoch: [%2d] [%4d/%4d] time: %4.4f" % (epoch, i, batch_idxs, time.time() - start_time)))
					counter += 1

