import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from baselines.a2c.utils import fc, conv_to_fc
from baselines.common.distributions import make_pdtype

from ppo2ttifrutti_agent import nenvs

class filmInit(object):
    def __init__(self, n):
        self.n = n
        initw1 = tf.constant_initializer([1.0]*32*(n+1))
        initb1 = tf.constant_initializer([0.0]*32*(n+1))

        # initw2 = tf.constant_initializer([1.0]*64*(n+1))
        # initb2 = tf.constant_initializer([0.0]*64*(n+1))

        # initw3 = tf.constant_initializer([1.0]*48*(n+1))
        # initb3 = tf.constant_initializer([0.0]*48*(n+1))

        self.film_w_1 = tf.get_variable('model/Film/w_1',32*(n+1),dtype=tf.float32, initializer=initw1)
        self.film_b_1 = tf.get_variable('model/Film/b_1',32*(n+1),dtype=tf.float32, initializer=initb1)
        # self.film_w_2 = tf.get_variable('model/Film/w_2',64*(n+1),dtype=tf.float32, initializer=initw2)
        # self.film_b_2 = tf.get_variable('model/Film/b_2',64*(n+1),dtype=tf.float32, initializer=initb2)
        # self.film_w_3 = tf.get_variable('model/Film/w_3',48*(n+1),dtype=tf.float32, initializer=initw3)
        # self.film_b_3 = tf.get_variable('model/Film/b_3',48*(n+1),dtype=tf.float32, initializer=initb3)

def CNN7(unscaled_images,index,filmObj):
    with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.contrib.layers.variance_scaling_initializer()):
        scaled_images = tf.cast(unscaled_images, tf.float32) / 255.
        activ = tf.nn.relu

        w_1 = tf.slice(filmObj.film_w_1,index*32,[32])
        b_1 = tf.slice(filmObj.film_b_1,index*32,[32])
        # w_2 = tf.slice(filmObj.film_w_2,index*64,[64])
        # b_2 = tf.slice(filmObj.film_b_2,index*64,[64])
        # w_3 = tf.slice(filmObj.film_w_3,index*48,[48])
        # b_3 = tf.slice(filmObj.film_b_3,index*48,[48])

        h = slim.separable_conv2d(scaled_images, 32, 8, 1, 4)
        # h = tf.math.add(tf.multiply(h, temp['weights_1']), temp['bias_1'])
        h = tf.math.add(tf.multiply(h, w_1), b_1)

        h2 = slim.separable_conv2d(h, 64, 4, 1, 2)
        # h2 = tf.math.add(tf.multiply(h2, temp['weights_2']), temp['bias_2'])
        # h2 = tf.math.add(tf.multiply(h2, w_2), b_2)
        
        h3 = slim.separable_conv2d(h2, 48, 3, 1, 1)
        # h3 = tf.math.add(tf.multiply(h3, temp['weights_3']), temp['bias_3'])
        # h3 = tf.math.add(tf.multiply(h3, w_3), b_3)

        h3 = conv_to_fc(h3)
        return activ(fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2)))

class CnnPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, filmObj, reuse=False,st = "act", **conv_kwargs): #pylint: disable=W0613
        nh, nw, nc = ob_space.shape
        ob_shape = (int(nbatch/nenvs), nh, nw, nc) # Use this
        self.pdtype = make_pdtype(ac_space)
        index = tf.placeholder(tf.int32,[1])
        X = tf.placeholder(tf.uint8, ob_shape) #obs
        with tf.variable_scope("model", reuse=reuse):
            h = CNN7(X,index,filmObj) #**conv_kwargs)
            vf = fc(h, 'v', 1)[:,0]
            self.pd, self.pi = self.pdtype.pdfromlatent(h, init_scale=0.01)
            print("Network:")
            [print(v.name, v.shape) for v in tf.trainable_variables()]
            print("Trainable variables:")
            print(np.sum([np.prod(v.get_shape()) for v in tf.trainable_variables()]))

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob,idx, *_args, **_kwargs):

            a, v, neglogp = sess.run([a0, vf, neglogp0], {X:ob, index:[idx]})
            
            return a, v, self.initial_state, neglogp

        def value(ob,idx, *_args, **_kwargs):
            # print('the shape of ob when value is called is ',ob.shape)

            return sess.run(vf, {X:ob, index:[idx]})

        self.X = X
        self.vf = vf
        self.step = step
        self.value = value
        self.index = index
