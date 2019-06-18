import tensorflow as tf
import os
import numpy as np
from utils import lrelu, nameop, tbn, obn

def conv(x, nfilt, name, padding='same', k=5, s=2, d=1, use_bias=True):
    return tf.layers.conv2d(x, filters=nfilt, kernel_size=k, padding=padding, strides=[s, s], dilation_rate=[d, d],
                            kernel_initializer=tf.truncated_normal_initializer(0, .02), activation=None,
                            use_bias=use_bias, name=name)

def conv_t(x, nfilt, name, padding='same', k=5, s=2, use_bias=True):
    return tf.layers.conv2d_transpose(x, filters=nfilt, kernel_size=k, padding=padding, strides=[s, s],
                            kernel_initializer=tf.random_normal_initializer(0, .02), activation=None,
                            use_bias=use_bias, name=name)

def unet_conv(x, nfilt, name, is_training, s=2, k=4, d=1, use_bias=True, batch_norm=None, activation=lrelu):
    x = conv(x, nfilt, name, use_bias=use_bias, d=d, s=s, k=k)
    if batch_norm:
        x = batch_norm(x, name='batch_norm_{}'.format(name), training=is_training)

    if activation:
        x = activation(x)
    return x

def unet_conv_t(x, encoderx, nfilt, name, is_training, skip_connections=True, s=2, k=4, use_bias=True, use_dropout=0, batch_norm=None, activation=tf.nn.relu):
    x = conv_t(x, nfilt, name, s=s, k=k, use_bias=use_bias)
    if use_dropout:
        x = tf.layers.dropout(x, use_dropout, training=is_training)

    if batch_norm:
        x = batch_norm(x, name='batch_norm_{}'.format(name), training=is_training)

    if activation:
        x = activation(x)

    if skip_connections:
        x = tf.concat([x, encoderx], 3)

    return x

def batch_normalization(tensor, name, training):
    return tf.contrib.layers.batch_norm(tensor, decay=0.9, updates_collections=None, is_training=training, scope=name, reuse=tf.AUTO_REUSE)
    # return tf.layers.batch_normalization(tensor, training=training, momentum=.9, scale=True, fused=True, name=name)

def adversarial_loss(logits, labels):

    return tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)

def siamese_lossfn(logits, labels=None, diff=False, diffmargin=10., samemargin=0.):
    if diff:
        return tf.maximum(0., diffmargin - tf.reduce_sum(logits, axis=-1))

    return tf.reduce_sum(logits, axis=-1)


class TravelGAN(object):
    def __init__(self,
        args,
        x1,
        x2,
        name='TwoGANs'):
        self.name = name
        self.args = args
        self.x1 = x1
        self.x2 = x2

        if self.args.restore_folder:
            self._restore(self.args.restore_folder, self.args.gpu_frac)
            return

        self.iteration = 0

        if self.x1 is not None and self.x2 is not None:
            self.datasetx1ph = tf.placeholder(tf.float32, x1.shape, name='datasetx1ph')
            self.datasetx2ph = tf.placeholder(tf.float32, x2.shape, name='datasetx2ph')
            datasetx1 = tf.data.Dataset.from_tensor_slices((self.datasetx1ph)).repeat().batch(args.batch_size)
            datasetx2 = tf.data.Dataset.from_tensor_slices((self.datasetx2ph)).repeat().batch(args.batch_size)
            self.iteratorx1 = datasetx1.make_initializable_iterator()
            self.iteratorx2 = datasetx2.make_initializable_iterator()

            self.iteratorx1.make_initializer(datasetx1, name='initializerx1')
            self.iteratorx2.make_initializer(datasetx2, name='initializerx2')

            x1ph = self.iteratorx1.get_next()
            x1ph = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), x1ph)

            x2ph = self.iteratorx2.get_next()
            x2ph = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), x2ph)

            cropdim = int(.8 * x1ph.get_shape()[1].value)
            x1ph = tf.map_fn(lambda img: tf.random_crop(img, [cropdim, cropdim, 3]), x1ph)
            x2ph = tf.map_fn(lambda img: tf.random_crop(img, [cropdim, cropdim, 3]), x2ph)
        else:
            x1ph = tf.placeholder(tf.float32, [None, args.imdim, args.imdim, 3], name='x1ph')
            x2ph = tf.placeholder(tf.float32, [None, args.imdim, args.imdim, 3], name='x2ph')

        self.xb1 = tf.placeholder_with_default(x1ph, shape=[None, args.imdim, args.imdim, 3], name='xb1')
        self.xb2 = tf.placeholder_with_default(x2ph, shape=[None, args.imdim, args.imdim, 3], name='xb2')

        self.lr = tf.placeholder(tf.float32, shape=[], name='lr')
        self.is_training = tf.placeholder(tf.bool, shape=[], name='is_training')

        self._build()
        self._build_loss()
        self._build_optimization()

        self.init_session(limit_gpu_fraction=self.args.gpu_frac)
        self.graph_init(self.sess)

    def init_session(self, limit_gpu_fraction=.4, no_gpu=False):
        if no_gpu:
            config = tf.ConfigProto(device_count={'GPU': 0})
            self.sess = tf.Session(config=config)
        elif limit_gpu_fraction:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=limit_gpu_fraction)
            config = tf.ConfigProto(gpu_options=gpu_options)
            self.sess = tf.Session(config=config)
        else:
            self.sess = tf.Session()

        if not self.args.restore_folder:
            self.sess.run(obn('initializerx1'), feed_dict={tbn('datasetx1ph:0'): self.x1})
            self.sess.run(obn('initializerx2'), feed_dict={tbn('datasetx2ph:0'): self.x2})

    def graph_init(self, sess=None):
        if not sess: sess = self.sess

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

        sess.run(tf.global_variables_initializer())

        return self.saver

    def save(self, iteration=None, saver=None, sess=None, folder=None):
        if not iteration: iteration = self.iteration
        if not saver: saver = self.saver
        if not sess: sess = self.sess
        if not folder: folder = self.save_folder

        savefile = os.path.join(folder, self.name)
        saver.save(sess, savefile, write_meta_graph=True)
        print("Model saved to {}".format(savefile))

    def _restore(self, restore_folder, limit_gpu_fraction, no_gpu=False):
        tf.reset_default_graph()
        self.init_session(limit_gpu_fraction, no_gpu)
        ckpt = tf.train.get_checkpoint_state(restore_folder)
        self.saver = tf.train.import_meta_graph('{}.meta'.format(ckpt.model_checkpoint_path))
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        if self.x1 and self.x2:
            self.sess.run(obn('initializerx1'), feed_dict={tbn('datasetx1ph:0'): self.x1})
            self.sess.run(obn('initializerx2'), feed_dict={tbn('datasetx2ph:0'): self.x2})
        self.iteration = 0
        print("Model restored from {}".format(restore_folder))

    def _build_loss(self):
        self._build_loss_D()
        self._build_loss_G()

    def _build_loss_D(self):
        """Discriminator loss."""
        self.loss_Dreal = .5 * tf.reduce_mean(adversarial_loss(self.D1_probs_z, tf.ones_like(self.D1_probs_z)))
        self.loss_Dreal += .5 * tf.reduce_mean(adversarial_loss(self.D2_probs_z, tf.ones_like(self.D2_probs_z)))

        self.loss_Dfake = .5 * tf.reduce_mean(adversarial_loss(self.D1_probs_G, tf.zeros_like(self.D1_probs_G)))
        self.loss_Dfake += .5 * tf.reduce_mean(adversarial_loss(self.D2_probs_G, tf.zeros_like(self.D2_probs_G)))

        self.loss_Dreal = nameop(self.loss_Dreal, 'loss_Dreal')
        tf.add_to_collection('losses', self.loss_Dreal)

        self.loss_Dfake = nameop(self.loss_Dfake, 'loss_Dfake')
        tf.add_to_collection('losses', self.loss_Dfake)

    def _build(self):
        self.G12 = Generator(args=self.args, name='G12')
        self.Gb2 = self.G12(self.xb1, is_training=self.is_training)
        self.Gb2 = nameop(self.Gb2, 'Gb2')

        self.G21 = Generator(args=self.args, name='G21')
        self.Gb1 = self.G21(self.xb2, is_training=self.is_training)
        self.Gb1 = nameop(self.Gb1, 'Gb1')

        self.D1 = Discriminator(args=self.args, name='D1')
        self.D2 = Discriminator(args=self.args, name='D2')

        self.D1_probs_z = self.D1(self.xb1, is_training=self.is_training)
        self.D1_probs_G = self.D1(self.Gb1, is_training=self.is_training)

        self.D2_probs_z = self.D2(self.xb2, is_training=self.is_training)
        self.D2_probs_G = self.D2(self.Gb2, is_training=self.is_training)

        self.S1 = SiameseNet(args=self.args, name='S1')

        self.S1_x = self.S1(self.xb1, 1, is_training=self.is_training)
        self.S1_G = self.S1(self.Gb2, 2, is_training=self.is_training)

        self.S2_x = self.S1(self.xb2, 2, is_training=self.is_training)
        self.S2_G = self.S1(self.Gb1, 1, is_training=self.is_training)

        self.S1_x = nameop(self.S1_x, 'S1_x')
        self.S1_G = nameop(self.S1_G, 'S1_G')
        self.S2_x = nameop(self.S2_x, 'S2_x')
        self.S2_G = nameop(self.S2_G, 'S2_G')

    def _build_loss_G(self):
        # fool the discriminator loss
        self.loss_G1_discr = tf.reduce_mean(adversarial_loss(self.D1_probs_G, tf.ones_like(self.D1_probs_G)))
        self.loss_G2_discr = tf.reduce_mean(adversarial_loss(self.D2_probs_G, tf.ones_like(self.D2_probs_G)))
        tf.add_to_collection('losses', nameop(self.loss_G1_discr, 'loss_G1_discr'))
        tf.add_to_collection('losses', nameop(self.loss_G2_discr, 'loss_G2_discr'))

        # siamese loss
        self.loss_S = tf.constant(0.)
        if self.args.lambda_siamese:
            orders = [np.array(list(range(i, self.args.batch_size)) + list(range(i))) for i in range(1, self.args.batch_size)]
            losses_S1 = []
            losses_S2 = []
            losses_S3 = []

            for i, order in enumerate(orders):
                other = tf.constant(order)

                dists_withinx1 = self.S1_x - tf.gather(self.S1_x, other)
                dists_withinx2 = self.S2_x - tf.gather(self.S2_x, other)
                dists_withinG1 = self.S1_G - tf.gather(self.S1_G, other)
                dists_withinG2 = self.S2_G - tf.gather(self.S2_G, other)

                losses_S1.append(tf.reduce_mean(siamese_lossfn((dists_withinx1)**2, diff=True)))
                losses_S1.append(tf.reduce_mean(siamese_lossfn((dists_withinx2)**2, diff=True)))

                losses_S2.append(tf.reduce_mean((dists_withinx1 - dists_withinG1)**2))
                losses_S2.append(tf.reduce_mean((dists_withinx2 - dists_withinG2)**2))

                losses_S3.append(tf.reduce_mean(tf.reduce_sum(-(tf.nn.l2_normalize(dists_withinx1, axis=[-1]) * tf.nn.l2_normalize(dists_withinG1, axis=[-1])), axis=-1)))
                losses_S3.append(tf.reduce_mean(tf.reduce_sum(-(tf.nn.l2_normalize(dists_withinx2, axis=[-1]) * tf.nn.l2_normalize(dists_withinG2, axis=[-1])), axis=-1)))


            self.loss_S1 = tf.reduce_mean(losses_S1)
            self.loss_S2 = tf.reduce_mean(losses_S2)
            self.loss_S3 = tf.reduce_mean(losses_S3)
            tf.add_to_collection('losses', nameop(self.loss_S1, 'loss_S1'))
            tf.add_to_collection('losses', nameop(self.loss_S2, 'loss_S2'))
            tf.add_to_collection('losses', nameop(self.loss_S3, 'loss_S3'))
            self.loss_S = self.loss_S1 + self.loss_S2 + self.loss_S3

        self.loss_G = self.args.lambda_adversary * (self.loss_G1_discr + self.loss_G2_discr)
        if self.args.lambda_siamese:
            self.loss_G += self.args.lambda_siamese * (self.loss_S)

    def _build_optimization(self):
        Gvars = [tv for tv in tf.global_variables() if 'G12' in tv.name or 'G21' in tv.name or 'S1' in tv.name]
        Dvars = [tv for tv in tf.global_variables() if 'D1' in tv.name or 'D2' in tv.name]

        optG = tf.train.AdamOptimizer(self.lr, beta1=.5)
        self.train_op_G = optG.minimize(self.loss_G, var_list=Gvars, name='train_op_G')

        optD = tf.train.AdamOptimizer(self.lr, beta1=.5)
        self.train_op_Dreal = optD.minimize(self.loss_Dreal, var_list=Dvars, name='train_op_Dreal')
        self.train_op_Dfake = optD.minimize(self.loss_Dfake, var_list=Dvars, name='train_op_Dfake')
        self.train_op_D = optD.minimize(.5 * self.loss_Dfake + .5 * self.loss_Dreal, var_list=Dvars, name='train_op_D')


    def get_layer(self, xb1, xb2, name=None):
        tensor_name = "{}:0".format(name)
        tensor = tbn(tensor_name)

        feed = {tbn('xb1:0'): xb1,
                tbn('xb2:0'): xb2,
                tbn('is_training:0'): False}

        layer = self.sess.run(tensor, feed_dict=feed)

        return layer

    def get_loss_names(self):
        losses = [tns.name[:-2].replace('loss_', '').split('/')[-1] for tns in tf.get_collection('losses')]
        return "Losses: {}".format(' '.join(losses))

    def get_loss(self, xb1, xb2):
        feed = {tbn('xb1:0'): xb1,
                tbn('xb2:0'): xb2,
                tbn('is_training:0'): False}

        losses = self.sess.run(tf.get_collection('losses'), feed_dict=feed)

        lstring = ' '.join(['{:.3f}'.format(loss) for loss in losses])

        return lstring

    def train(self, x1=None, x2=None):
        self.iteration += 1

        feed = {tbn('lr:0'): self.args.learning_rate,
                tbn('is_training:0'): True}
        if x1 is not None:
            feed[tbn('xb1:0')] = x1
            feed[tbn('xb2:0')] = x2


        self.sess.run([obn('train_op_D')], feed_dict=feed)
        self.sess.run([obn('train_op_G')], feed_dict=feed)



class SiameseNet(object):
    def __init__(self,
        args,
        name=''):
        self.args = args
        self.name = name
        self.first_call = True

    def __call__(self, x, domain, is_training):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            if self.first_call: print(tf.get_variable_scope().name)

            # down
            nshape = x.get_shape()[1].value
            nfilt = self.args.nfilt * 1
            layer = 1
            iinput = x
            minshape = 4
            maxfilt = 8 * nfilt

            while nshape > minshape:
                # pre
                if self.first_call: print(iinput)
                # layer
                if (layer == 1):
                    layer_name = "{}_{}".format(layer, domain)
                else:
                    layer_name = layer
                output = unet_conv(iinput, nfilt, 'h{}'.format(layer_name), is_training, activation=lrelu, batch_norm=batch_normalization if layer != 1 else False)
                # post
                nshape /= 2
                nfilt = min(2 * nfilt, maxfilt)
                layer += 1
                iinput = output
            if self.first_call: print(output)

            output = tf.layers.flatten(output)

            out = tf.layers.dense(output, self.args.siamese_latentdim, name='out')

            if self.first_call: print("{}\n".format(out))
            self.first_call = False
        return out

class Generator(object):
    def __init__(self,
        args,
        name=''):
        self.args = args
        self.name = name
        self.first_call = True

    def __call__(self, x, is_training):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            if self.first_call: print(tf.get_variable_scope().name)

            # up
            encoders = []
            nshape = x.get_shape()[1].value
            filt = self.args.nfilt
            layer = 1
            iinput = x
            minshape = 8
            basefilt = filt
            maxfilt = 4 * filt
            nfilt = basefilt

            while nshape > minshape:
                # pre
                if self.first_call: print(iinput)
                # layer
                output = unet_conv(iinput, nfilt, 'e{}'.format(layer), is_training, activation=lrelu, batch_norm=batch_normalization if layer != 1 else False)
                # post
                encoders.append([output, nshape, nfilt, layer])
                nshape /= 2
                nfilt = min(2 * nfilt, maxfilt)
                layer += 1
                iinput = output

            if self.first_call: print(output)

            iinput = encoders.pop()[0]

            # down
            for encoderinput, nshape, nfilt, layer in encoders[::-1]:
                # layer
                output = unet_conv_t(iinput, encoderinput, nfilt, 'd{}'.format(layer), is_training, activation=tf.nn.relu, batch_norm=batch_normalization)
                # post
                iinput = output
                if self.first_call: print(iinput)

            # out
            out = unet_conv_t(output, None, 3, 'out', is_training, activation=tf.nn.tanh, batch_norm=False, skip_connections=False)

            if self.first_call:
                print("{}\n".format(out))
                self.first_call = False

        return out

class Discriminator(object):

    def __init__(self,
        args,
        name=''):
        self.args = args
        self.name = name
        self.first_call = True

    def __call__(self, x, is_training):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            if self.first_call: print(tf.get_variable_scope().name)

            minshape = 4
            filt = self.args.nfilt * 2
            maxfilt = 8 * filt
            basefilt = filt
            nfilt = basefilt
            nshape = x.get_shape()[1].value
            layer = 1
            iinput = x

            while nshape > minshape:
                if self.first_call: print(iinput)
                output = unet_conv(iinput, nfilt, 'h{}'.format(layer), is_training, activation=lrelu, batch_norm=batch_normalization if layer != 1 else False)
                nshape /= 2
                nfilt = min(2 * nfilt, maxfilt)
                layer += 1
                iinput = output
            if self.first_call: print(output)

            output = tf.layers.flatten(output)
            out = tf.layers.dense(output, 1, kernel_initializer=tf.random_normal_initializer(0, .02), name='out')

            if self.first_call:
                print("{}\n".format(out))
                self.first_call = False

        return out











































