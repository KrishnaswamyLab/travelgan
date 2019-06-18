import matplotlib
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import argparse
from skimage.transform import resize as imresize
import scipy.ndimage
from utils import now
from travelgan import TravelGAN
from loader import Loader

def get_data_args(args):
    batch1, batch2, args.channels, args.imdim = get_data_imagenet(args.datadirb1, args.datadirb2, D=int(1.25 * args.downsampledim))
    return args, batch1, batch2

def get_data_imagenet(datadirb1, datadirb2, D=128):
    b1 = sorted(glob.glob('{}/*'.format(datadirb1)))
    b2 = sorted(glob.glob('{}/*'.format(datadirb2)))
    b1 = [fn for fn in b1 if any(['png' in fn.lower(), 'jpeg' in fn.lower(), 'jpg' in fn.lower()])]
    b2 = [fn for fn in b2 if any(['png' in fn.lower(), 'jpeg' in fn.lower(), 'jpg' in fn.lower()])]

    b1 = [scipy.misc.imresize(scipy.ndimage.imread(f), (D, D)) for f in b1]
    b2 = [scipy.misc.imresize(scipy.ndimage.imread(f), (D, D)) for f in b2]
    b1 = [im for im in b1 if len(im.shape) == 3]
    b2 = [im for im in b2 if len(im.shape) == 3]

    b1 = np.stack(b1, axis=0)
    b2 = np.stack(b2, axis=0)

    b1 = b1.astype(np.float32)
    b2 = b2.astype(np.float32)
    print(b1.shape)
    print(b2.shape)

    b1 = (b1 / 127.5) - 1
    b2 = (b2 / 127.5) - 1

    return b1, b2, 3, int(.8 * D)

def randomize_image(img, enlarge_size=286, output_size=256):
    img = imresize(img, [enlarge_size, enlarge_size])

    h1 = int(np.ceil(np.random.uniform(1e-2, enlarge_size - output_size)))
    w1 = int(np.ceil(np.random.uniform(1e-2, enlarge_size - output_size)))
    img = img[h1:h1 + output_size, w1:w1 + output_size]

    if np.random.random() > .5:
        img = np.fliplr(img)

    return img

def randomcrop(imgs, cropsize):
    imgsout = np.zeros((imgs.shape[0], cropsize, cropsize, imgs.shape[3]))
    for i in range(imgs.shape[0]):
        img = imgs[i]
        h1 = int(np.ceil(np.random.uniform(1e-2, img.shape[1] - cropsize)))
        w1 = int(np.ceil(np.random.uniform(1e-2, img.shape[1] - cropsize)))
        img = img[h1:h1 + cropsize, w1:w1 + cropsize]
        if np.random.random() > .5:
            img = np.fliplr(img)
        imgsout[i] = img
    return imgsout

def parse_args():
    parser = argparse.ArgumentParser()
    # required params
    parser.add_argument('--savefolder', type=str)

    # data params
    parser.add_argument('--downsampledim', type=int, default=128)
    parser.add_argument('--datadirb1', type=str, default='')
    parser.add_argument('--datadirb2', type=str, default='')

    # model params
    parser.add_argument('--nfilt', type=int, default=64)
    parser.add_argument('--lambda_adversary', type=float, default=1)

    # siamese params
    parser.add_argument('--lambda_siamese', type=float, default=10)
    parser.add_argument('--siamese_latentdim', type=int, default=1000)

    # training params
    parser.add_argument('--training_steps', type=int, default=200000)
    parser.add_argument('--training_steps_decayafter', type=int, default=200000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--gpu_frac', type=float, default=.4)
    parser.add_argument('--restore_folder', type=str, default='')
    parser.add_argument('--learning_rate', type=float, default=.0002)

    args = parser.parse_args()

    args.modelname = 'TraVeLGAN'
    args.model = TravelGAN
    args, batch1, batch2 = get_data_args(args)

    if not os.path.exists(args.savefolder): os.mkdir(args.savefolder)

    return args, batch1, batch2

args, batch1, batch2 = parse_args()
if not args.restore_folder:
    with open(os.path.join(args.savefolder, 'args.txt'), 'w+') as f:
        for arg in vars(args):
            argstring = "{}: {}\n".format(arg, vars(args)[arg])
            f.write(argstring)
            print(argstring[:-1])

if not os.path.exists("{}/output".format(args.savefolder)): os.mkdir("{}/output".format(args.savefolder))


load1 = Loader(batch1, labels=np.arange((batch1.shape[0])), shuffle=True)
load2 = Loader(batch2, labels=np.arange((batch2.shape[0])), shuffle=True)

print("Domain 1 shape: {}".format(batch1.shape))
print("Domain 2 shape: {}".format(batch2.shape))
model = args.model(args, x1=batch1, x2=batch2, name=args.modelname)


plt.ioff()
fig = plt.figure(figsize=(4, 10))
np.set_printoptions(precision=3)
decay = model.args.learning_rate / (args.training_steps - args.training_steps_decayafter)

for i in range(1, args.training_steps):

    if i % 10 == 0: print("Iter {} ({})".format(i, now()))
    model.train()

    if i >= args.training_steps_decayafter:
        model.args.learning_rate -= decay

    if i and (i == 50 or i % 500 == 0):
        model.save(folder=args.savefolder)

        xb1inds = np.random.choice(batch1.shape[0], replace=False, size=[10])
        xb2inds = np.random.choice(batch2.shape[0], replace=False, size=[10])
        testb1 = batch1[xb1inds]
        testb2 = batch2[xb2inds]

        testb1 = randomcrop(testb1, args.imdim)
        testb2 = randomcrop(testb2, args.imdim)

        Gb1 = model.get_layer(testb1, testb2, name='Gb1')
        Gb2 = model.get_layer(testb1, testb2, name='Gb2')

        # back to [0,1] for imshow
        testb1 = (testb1 + 1) / 2
        testb2 = (testb2 + 1) / 2
        Gb1 = (Gb1 + 1) / 2
        Gb2 = (Gb2 + 1) / 2

        fig.clf()
        fig.subplots_adjust(.01, .01, .99, .99, .01, .01)
        for ii in range(10):
            ax1 = fig.add_subplot(10, 2, 2 * ii + 1)
            ax2 = fig.add_subplot(10, 2, 2 * ii + 2)
            for ax in (ax1, ax2):
                ax.set_xticks([])
                ax.set_yticks([])

            ax1.imshow(testb1[ii])
            ax2.imshow(Gb2[ii])
        fig.canvas.draw()
        fig.savefig('{}/output/b1_to_b2.png'.format(args.savefolder), dpi=500)

        fig.clf()
        fig.subplots_adjust(.01, .01, .99, .99, .01, .01)
        for ii in range(10):
            ax1 = fig.add_subplot(10, 2, 2 * ii + 1)
            ax2 = fig.add_subplot(10, 2, 2 * ii + 2)
            for ax in (ax1, ax2):
                ax.set_xticks([])
                ax.set_yticks([])

            ax1.imshow(testb2[ii])
            ax2.imshow(Gb1[ii])
        fig.canvas.draw()
        fig.savefig('{}/output/b2_to_b1.png'.format(args.savefolder), dpi=500)


        xb1inds = np.random.choice(batch1.shape[0], replace=False, size=[args.batch_size])
        xb2inds = np.random.choice(batch2.shape[0], replace=False, size=[args.batch_size])
        testb1 = batch1[xb1inds]
        testb2 = batch2[xb2inds]

        testb1 = randomcrop(testb1, args.imdim)
        testb2 = randomcrop(testb2, args.imdim)

        print(model.get_loss_names())
        lstring = model.get_loss(testb1, testb2)
        print("{} ({}): {}".format(i, now(), lstring))











