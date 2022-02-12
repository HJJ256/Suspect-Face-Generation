""" predict_feature labels of synthetic_images """

import os
import glob
import numpy as np
import PIL.Image
import h5py
import tensorflow as tf

# path to model generated results
path_gan_sample_img = './asset_results/sgan2_ffhq_sample_jpg/'
file_pattern_x = 'sample_*.jpg'
file_pattern_z = 'sample_*_z.npy'
filename_sample_y = 'sample_y.h5'

while os.path.basename(os.getcwd()) in ('tl_gan', 'src'):
    os.chdir('..')
assert ('README.md' in os.listdir('./')), 'Can not find project root, please cd to project root before running the following code'

print(os.getcwd())
import cnn_face_attr_celeba as cnn_face

# get the list of image_names
list_pathfile_x = glob.glob(os.path.join(path_gan_sample_img, file_pattern_x))
list_pathfile_z = glob.glob(os.path.join(path_gan_sample_img, file_pattern_z))
list_pathfile_x.sort()
list_pathfile_z.sort()
cnn_face
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
allow_growth_session = tf.Session(config=config)
tf.keras.backend.set_session(allow_growth_session)

assert len(list_pathfile_x) == len(list_pathfile_z), 'num_image does not match num_z'

##
""" load model for prediction """
model = cnn_face.create_cnn_model()
model.load_weights(cnn_face.get_list_model_save()[-1])

list_y = []
batch_size = 64
list_img_batch = []
list_pathfile_x_use = list_pathfile_x
num_use = len(list_pathfile_x_use)
save_every = 2048

for i, pathfile_x in enumerate(list_pathfile_x_use):
    img = np.asarray(PIL.Image.open(pathfile_x))
    img = img.astype(np.float32)
    list_img_batch.append(img)

    if i % batch_size == batch_size-1 or i==num_use-1:
        print('{}/{}'.format(i+1, num_use))
        img_batch = np.stack(list_img_batch, axis=0)
        x = cnn_face.preprocess_input(img_batch)
        y = model.predict(x, batch_size=batch_size)
        list_y.append(y)

        list_img_batch = []

        if i % save_every == 0:
            y_concat = np.concatenate(list_y, axis=0)
            pathfile_sample_y = os.path.join(path_gan_sample_img, filename_sample_y)
            with h5py.File(pathfile_sample_y, 'w') as f:
                f.create_dataset('y', data=y_concat)


y_concat = np.concatenate(list_y, axis=0)
pathfile_sample_y = os.path.join(path_gan_sample_img, filename_sample_y)
with h5py.File(pathfile_sample_y, 'w') as f:
    f.create_dataset('y', data=y_concat)

