"""
try face tl_gan using pg-gan model, modified from
https://drive.google.com/drive/folders/1A79qKDTFp6pExe4gTSgBsEOkxwa2oes_
"""

"""
prerequsit: before running the code, download pre-trained model to project_root/asset_model/
pretrained model download url: https://drive.google.com/drive/folders/15hvzxt_XxuokSmj0uO4xxMTMWVc0cIMU
model name: karras2018iclr-celebahq-1024x1024.pkl
"""

import os
import sys
import time
import pickle
import numpy as np
import tensorflow as tf
import PIL.Image
import datetime

while os.path.basename(os.getcwd()) in ('tl_gan', 'src'):
    os.chdir('..')
assert ('README.md' in os.listdir('./')), 'Can not find project root, please cd to project root before running the following code'

# path to model code and weight
path_pg_gan_code = './src/model/pggan'
path_model = './asset_model/stylegan2-ffhq-config-f.pkl'
sys.path.append(path_pg_gan_code)

# path to model generated results
path_gen_sample = './asset_results/sgan2_ffhq_sample_pkl/'
if not os.path.exists(path_gen_sample):
    os.mkdir(path_gen_sample)
path_gan_explore = './asset_results/sgan2_ffhq_explore/'
if not os.path.exists(path_gan_explore):
    os.mkdir(path_gan_explore)


""" gen samples and save as pickle """

n_batch = 32000
batch_size = 8

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
allow_growth_session = tf.Session(config=config)
tf.keras.backend.set_session(allow_growth_session)

with tf.Session() as sess:

    # Import official CelebA-HQ networks.
    try:
        with open(path_model, 'rb') as file:
            G, D, Gs = pickle.load(file)
    except FileNotFoundError:
        print('before running the code, download pre-trained model to project_root/asset_model/')
        raise

    # Generate latent vectors.
    # latents = np.random.RandomState(1000).randn(1000, *Gs.input_shapes[0][1:]) # 1000 random latents
    # latents = latents[[477, 56, 83, 887, 583, 391, 86, 340, 341, 415]] # hand-picked top-10

    for i_batch in range(n_batch):
        try:
            i_sample = i_batch * batch_size

            tic = time.time()

            latents = np.random.randn(batch_size, *Gs.input_shapes[0][1:])
            labels = np.zeros([latents.shape[0]] + Gs.input_shapes[1][1:])
            dlatents = Gs.components['mapping'].run(latents,labels)
            dl_mean = np.mean(dlatents,axis = 0)
            dl_mean = dlatents + (dl_mean-dlatents)*0.995
            dl_psi = dl_mean + (dl_mean-dlatents)*0.5

            # Generate dummy labels (not used by the official networks).
            

            # Run the generator to produce a set of images.
            images = Gs.components['synthesis'].run(dl_psi)

            images = np.clip(np.rint((images + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8)  # [-1,1] => [0,255]
            images = images.transpose(0, 2, 3, 1)  # NCHW => NHWC

            images = images[:, 4::8, 4::8, :]
            #print(images.shape)

            with open(os.path.join(path_gen_sample, 'sgan2_celeba_{:0>6d}.pkl'.format(i_sample)), 'wb') as f:
                pickle.dump({'z':latents,'w': dl_psi[:,0], 'x': images}, f)

            toc = time.time()
            print(i_sample, toc-tic)

        except:
            print('error in {}'.format(i_sample))


""" view generated samples """
yn_view_sample = True
if yn_view_sample:
    with open(os.path.join(path_gen_sample, 'sgan2_celeba_{:0>6d}.pkl'.format(0)), 'rb') as f:
        temp = pickle.load(f)

    import matplotlib.pyplot as plt
    plt.imshow(temp['x'][0]); plt.show()


