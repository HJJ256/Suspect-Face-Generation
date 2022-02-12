import os
import glob
import sys
import numpy as np
import pickle
import tensorflow as tf
import PIL
import ipywidgets
import io
import gc
import time
from PIL import Image
""" make sure this notebook is running from root directory """
while os.path.basename(os.getcwd()) in ('notebooks', 'src','tl_gan'):
    os.chdir('..')
print(os.getcwd())
assert ('README.md' in os.listdir('./')), 'Can not find project root, please cd to project root before running the following code'
#import src.tl_gan.generate_image as generate_image
#import src.tl_gan.feature_axis as feature_axis
#import src.tl_gan.feature_celeba_organize as feature_celeba_organize
print(os.listdir())
from dnnlib.tflib import init_tf,run
from sgantraining import misc

import h5py

path_gan_sample_img = './asset_results/stylegan_celeba_sample_jpg/'
file_pattern_x = 'sample_*.jpg'
file_pattern_z = 'sample_*_z.npy'
filename_sample_y = 'sample_y_000celebahq-classifier-05-5-o-clock-shadow.h5'

list_pathfile_x = glob.glob(os.path.join(path_gan_sample_img, file_pattern_x))
list_pathfile_z = glob.glob(os.path.join(path_gan_sample_img, file_pattern_z))
list_pathfile_x.sort()
list_pathfile_z.sort()

batch_size = 16
img_batch = np.zeros((16,256,256,3),dtype=np.float32)
list_pathfile_x_use = list_pathfile_x
num_use = len(list_pathfile_x_use)
save_every = 2048
y_concat_temp = None
y_concat = None
init_tf()
clf = misc.load_pkl('asset_model/stylegan_face_attr_classifiers/000celebahq-classifier-05-5-o-clock-shadow.pkl')
results = np.zeros((len(list_pathfile_x_use)//batch_size,batch_size,1))
factor = 1
graph1 = tf.Graph()
with graph1.as_default():
    img_batc = tf.placeholder(dtype='float32',shape=(16,256,256,3))
    img_batch_tt = tf.reshape(img_batc, [-1, img_batc.shape[3], img_batc.shape[1] // factor, factor, img_batc.shape[2] // factor, factor])
    img_batch_tt = tf.reduce_mean(img_batch_tt, axis=[3, 5])
        
j = 0


for i, pathfile_x in enumerate(list_pathfile_x_use):
    #img = (np.asarray(PIL.Image.open(pathfile_x))/255).astype(np.float32)
    #img = img.astype(np.float32)
    with open(pathfile_x,'rb') as f: 
        img_batch[i%batch_size] = (np.asarray(PIL.Image.open(f))/255).astype(np.float32)
    #img[:] = None
    if i % batch_size == batch_size-1 or i==num_use-1:
        print('{}/{}'.format(i+1, num_use))
        #factor = img_batch.shape[1] // 256
        #print(sys.getsizeof(img_batch))
        #print(sys.getsizeof(results))
        with tf.Session(graph=graph1) as sess:
            #sess.run(init)
            img_batch_t = sess.run(img_batch_tt,feed_dict={img_batc:img_batch})
        #logits = 
        results[j] = run(clf.get_output_for(img_batch_t,None))
        j+=1
        #print(results[0].shape)
        del img_batch_t
        #del logits
        gc.collect()
        #print(str(sys.getsizeof(results)),len(results))
        #if (i+1)%save_every == 0:
        #    print('hello')
        #    
        #    y_concat_temp = np.concatenate(results,axis=0)
        #    
        #    if len(y_concat_temp)==num_use:
        #        if y_concat is None:
        #            y_concat = np.concatenate([y_concat_temp],axis=0)
        #        else:
        #            y_concat = np.concatenate((y_concat,y_concat_temp),axis=1)
        #        
        #    if y_concat is not None:
        #        pathfile_sample_y = os.path.join(path_gan_sample_img, filename_sample_y)
        #        with h5py.File(pathfile_sample_y, 'w') as f:
        #            f.create_dataset('y', data=y_concat)

y_concat = np.concatenate(results,axis=0)
                                  
if y_concat is not None:
    pathfile_sample_y = os.path.join(path_gan_sample_img, filename_sample_y)
    with h5py.File(pathfile_sample_y, 'w') as f:
        f.create_dataset('y', data=y_concat)
