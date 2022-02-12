from django.shortcuts import render
from django.http import HttpResponse
# Create your views here.
from collections import Counter
import pandas as pd
from PIL import Image
import os
import imageio
from django.core.cache import cache
from django.views.decorators.cache import never_cache
from .models import Image_database
from django.views.decorators.clickjacking import xframe_options_exempt

import glob
import sys
import numpy as np
import pickle
import tensorflow as tf
import ipywidgets
import io

from tl_gan import generate_image, feature_axis, feature_celeba_organize
from tl_gan import sg2projector as projector
from dnnlib.tflib import init_tf

sess = None
Gss = None
w_lats = None
value = None
proj = None

def load_feature_directions(path_feature_direction = './asset_results/sgan2_ffhq_feature_direction_40/'):
    """ load feature directions """
    pathfile_feature_direction = glob.glob(os.path.join(path_feature_direction, 'feature_direction_*.pkl'))[-1]

    with open(pathfile_feature_direction, 'rb') as f:
        feature_direction_name = pickle.load(f)

    feature_direction = feature_direction_name['direction']
    feature_name = feature_direction_name['name']
    num_feature = feature_direction.shape[1]

    import importlib
    importlib.reload(feature_celeba_organize)
    feature_name = feature_celeba_organize.feature_name_celeba_rename
    feature_direction = feature_direction_name['direction']* feature_celeba_organize.feature_reverse[None, :]
    feature_lock_status = np.zeros(num_feature).astype('bool')
    feature_direction_disentangled = feature_axis.disentangle_feature_axis_by_idx(
        feature_direction, idx_base=np.flatnonzero(feature_lock_status))
        
    #print(list(range(num_feature)))
    #print(feature_name)
    #print(feature_lock_status.shape)
    return feature_name,feature_direction,feature_lock_status,feature_direction_disentangled


def modify_along_feature(latents, idx_feature, feature_direction_disentangled, feature_lock_status, step_size=0.01):
    idx_base = np.flatnonzero(feature_lock_status)
    if len(idx_base)== 0:
        latents += feature_direction_disentangled[:, idx_feature] * step_size
    else:
        disent = feature_direction_disentangled[:, idx_feature].reshape(512,1)
        v = 0
        #print(len(idx_base))
        for idx in idx_base:
            #print(idx)
            di = feature_direction_disentangled[:,idx].reshape(512,1)
            v += np.dot(disent.T,di)*di
            #print((np.dot(disent.T,di)*di).shape)
        disent = disent - v
        latents +=  disent.reshape(512) * step_size
        
    #print(feature_direction_disentangled[:,idx_feature].reshape(512,1).T.shape)

def set_feature_lock(feature_direction_disentangled, feature_lock_status, idx_feature, set_to=None):
    if set_to is None:
        feature_lock_status[idx_feature] = np.logical_not(feature_lock_status[idx_feature])

    else:
        feature_lock_status[idx_feature] = set_to
    feature_direction_disentangled = feature_axis.disentangle_feature_axis_by_idx(
        feature_direction, idx_base=np.flatnonzero(feature_lock_status))
    #print(feature_lock_status)
        
def update_img(lats):        
    #x_sample = generate_image.gen_single_img(z=latents, Gs=Gs)
    x_sample = Gss.run(lats)
    x_sample = np.clip(np.rint((x_sample + 0.0) / 1.0 * 255.0), 0.0, 255.0).astype(np.uint8)  # [-1,1] => [0,255]
    x_sample = x_sample.transpose(0, 2, 3, 1)[0]  # NCHW => NHWC
    Image.fromarray(x_sample).save(os.path.join(settings.MEDIA_ROOT, 'tmp.png'),format='PNG')
    img_dat = Image_database()
    img_dat.id = 0
    img_dat.photo.name = 'tmp.png'
    img_dat.save()


        
def initialize_projector(path_model = './asset_model/stylegan2-ffhq-config-f.pkl'):
    """ create tf session """
    yn_CPU_only = False
    proj = projector.Projector()
    
    if yn_CPU_only:
        config = tf.ConfigProto(device_count = {'GPU': 0}, allow_soft_placement=True)
    else:
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

    sess = tf.InteractiveSession(config=config)

    try:
        with open(path_model, 'rb') as file:
            _, _, Gs = pickle.load(file)
    except FileNotFoundError:
        print('before running the code, download pre-trained model to project_root/asset_model/')
        raise

    proj.set_network(Gs)

    del Gs
    
    return sess,proj
    

def project_image(proj, targets, num_steps):
    proj.num_steps = num_steps
    proj.start(targets)
    while proj.get_cur_step() < proj.num_steps:
        print('\r%d / %d ... ' % (proj.get_cur_step(), proj.num_steps), end='', flush=True)
        proj.step()
    print('\r%-30s\r' % '', end='', flush=True)
    return proj.get_dlatents()

features = ['5_o_Clock_Shadow', 'Arched_Eyebrows',
    'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips',
    'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair',
    'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee',
    'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male',
    'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
    'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
    'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair',
    'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
    'Wearing_Necklace', 'Wearing_Necktie', 'Young']

male_second_set = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Bangs', 'Big_Lips',
       'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Brown_Hair',
       'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Goatee', 'Gray_Hair',
       'High_Cheekbones', 'Mustache', 'Narrow_Eyes',
       'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
       'Rosy_Cheeks', 'Sideburns', 'Straight_Hair', 'Wavy_Hair', 'Young']

female_first_set = ['Arched_Eyebrows', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair',
       'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby',
       'Double_Chin', 'Gray_Hair', 'High_Cheekbones',
       'Narrow_Eyes', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose',
       'Receding_Hairline', 'Rosy_Cheeks', 'Straight_Hair', 'Wavy_Hair',
       'Young']

male_features = ['5_o_Clock_Shadow','Bald','Goatee','Mustache','Sideburns','No_Beard']
dont_ask = ['Blurry','Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie','Attractive',
 'Bags_Under_Eyes','Mouth_Slightly_Open','Heavy_Makeup','Smiling','Eyeglasses']
is_ones = 'Attractive,Chubby,Smiling,Young,Bald'

feature_name,feature_direction,feature_lock_status,feature_direction_disentangled = load_feature_directions()

morph_features = ['Male','Age','Skin_Tone','Bangs','Hairline','Bald','Big_Nose','Pointy_Nose','Makeup','Smiling','Mouth_Open','Wavy_Hair',
    'Beard','Goatee','Sideburns','Blond_Hair','Black_Hair','Gray_Hair','Eyeglasses','Earrings','Necktie']

    
features_idx = [feature_name.index(i) for i in morph_features]
feature_direction = feature_direction[:,features_idx]
feature_direction_disentangled = feature_direction_disentangled[:,features_idx]
feature_lock_status = feature_lock_status[features_idx]
    
initial_input = {}

value = None

def index(request):
    cache.clear()
    return render(request, "index.html")

def image(request):
    img_dat = Image_database.objects.get(id = 0)
    return render(request,"image.html",{'img_dat':img_dat})

@xframe_options_exempt
def image_morph(request):
    global sess
    global value
    global Gss
    global w_lats
    global proj
    value = request.POST['test']
    if sess is None:    
        sess,proj = initialize_projector()
    if Gss is None:
        Gss = proj._Gs.components['synthesis']
    image = Image.open(os.path.join(settings.BASE_DIR,'static/png/' + value ))
    img_arr = np.array(image,dtype='float32').reshape(1,1024,1024,3)
    img_arr = np.transpose(img_arr,(0,3,1,2))
    w_lats = project_image(proj,img_arr/255.0,500)
    del(image)
    del(img_arr)
    x_sample = Gss.run(w_lats)
    x_sample = np.clip(np.rint((x_sample + 0.0) * 255 / 1.0), 0.0, 255.0).astype(np.uint8)  # [-1,1] => [0,255]
    x_sample = x_sample.transpose(0, 2, 3, 1)[0]
    Image.fromarray(x_sample).save(os.path.join(settings.MEDIA_ROOT, 'tmp.png'),format='PNG')
    img_dat = Image_database()
    img_dat.id = 0
    img_dat.photo.name = 'tmp.png'
    img_dat.save()
    print(img_dat.photo.path, img_dat.photo.url, img_dat.id)

    
    return render(request,"morphine.html" , {'val': value , 'morph_features' : morph_features,'img_dat':Image_database.objects.get(id = 0),'features_idx':features_idx})




from django.conf import settings
import pandas as pd
import numpy as np
from PIL import Image
import io

def morphine(request):
    print(request.FILES)
    temp = pd.read_csv(os.path.join(settings.BASE_DIR, 'test.csv'))
    print("----->",value)
    #img = Image.open(os.path.join(settings.BASE_DIR,"C:/Users/91982/Downloads/sfg/sfg/static/images/"+value))
    
    step_size = 0.3
    for i in range(1, 22): #change this to button press logic
        txt = temp['id'][i - 1]
        print(txt)

        if 'plus'+str(i) in request.POST:
            modify_along_feature(w_lats, i-1, feature_direction_disentangled, feature_lock_status, step_size=1*step_size)
            update_img(w_lats)
        elif 'minus'+str(i) in request.POST:
            modify_along_feature(w_lats, i-1, feature_direction_disentangled, feature_lock_status, step_size=-1*step_size)
            update_img(w_lats)
        elif 'feat'+str(i) in request.POST:
            set_feature_lock(feature_direction_disentangled, feature_lock_status, i-1)
            print(feature_lock_status)
            #update_img(w_lats)

        temp['id'][i-1] = txt

    temp.to_csv(os.path.join(settings.BASE_DIR, 'test.csv'), index= False ,header = 'False')

    temp = pd.read_csv(os.path.join(settings.BASE_DIR, 'test.csv'))
    
    for i in temp['id']:
        print("updated value...... ",i)
   
    response = HttpResponse('sss')
    return response


def edit(request):
    return render(request,"edit.html", {'val': 'imgHQ02812.png'})

def after_male(request):
    male = request.POST["male"]
    if(male == "yes"):
        initial_input['Male'] = 1
        next_ques_set = ['Beard', 'Bald']
        return render(request, "male_set1.html",{'features' : next_ques_set , 'radcount' : ['yes','no'], 'is_ones' : is_ones})
    elif(male == "no"):
        initial_input['Male'] = -1
        for i in male_features:
            initial_input[i] = -1
        return render(request, "female_set.html",{'features' : female_first_set , 'radcount' : ['yes','no'], 'is_ones' : is_ones})
    
"""
    if(initial_input['Male'] == -1):
        initial_input['No_Beard'] = 1
        for i in male_features:
            features = [i for i in features if i not in male_features]
            initial_input[i] = -1
    else:
        is_ones.appf1end('Bald')
    
    next_ques = ['Bald', 'No_Beard']

    features.remove('Male')
""" 


def dat_image(datfiles):
    ls = ['imgHQ02516.png', 'imgHQ02532.png', 'imgHQ02551.png','imgHQ02645.png','imgHQ02666.png','imgHQ02812.png',
    'imgHQ02707.png','imgHQ02810.png','imgHQ03345.png']         

    # remove the above list when actual database of images is present
    #for files in datfiles:
    #    new = "imgHQ" + files[5:10] + ".png"
    #    ls.append(new)
    return ls


def male_set2(request):
    beard = request.POST["Beard"]
    bald = request.POST["Bald"]
    initial_input['No_Beard'] = -1 if beard == 'yes' else 1
    initial_input['Bald'] = 1 if bald == 'yes' else -1
    if(initial_input['Bald'] == 1):
        for i in ['Bangs','Straight_Hair','Wavy_Hair']:
            male_second_set.remove(i)
    if(initial_input['No_Beard'] == 1 and initial_input['Bald'] == 1):
        for i in ['Black_Hair','Blond_Hair','Brown_Hair','Gray_Hair']:
            male_second_set.remove(i)
    return render(request,"male_set2.html" , {'features' : male_second_set , 'radcount' : ['yes','no'] , 'is_ones' : is_ones})


def f_show_image_option(request):
    for i in female_first_set:
        val = request.POST[i]
        if(val == "yes"):
            initial_input[i] = 1
        elif(val == "no"):
            initial_input[i] = -1

    df = pd.read_csv("celeba_hq_dataset.csv")
    df = df.drop(df.index[len(df)-1])
    features = list(df)[1:]

    a = initial_input['Male']
    df = df[df['Male'] == a]
    count = len(df)
    while(len(df)>5):
        prev_df = len(df)
        pblts = {}
        total_rows = len(df)
        for a in features:
            if(a not in dont_ask):
                temp = Counter(df[a])
                pblts[a] = {-1 : round(temp[-1]/total_rows,4) , 1 : round(temp[1]/total_rows,2)} 

        get_new_df = {}
        for k,v in initial_input.items():
            temp = pblts[k]
            if(pblts[k] not in dont_ask): 
                get_new_df[k] = [v,temp[v]]

        min = 999
        for k,v in get_new_df.items():
            t1 = v
            if(t1[1] < min):
                min = t1[1]
                split_feature = k
                split_value = t1[0]

        df = df[df[split_feature] == split_value]
        if(prev_df == len(df)):
            break
    x = list(df['index'])
    print(x)
    #x = dat_image(x)
    return render(request,"show_image_option.html" , {'initial_input': initial_input,'df' : x , 'count' : count})
    
def m_show_image_option(request):
    print(request.FILES)
    for i in male_second_set:
        val = request.POST[i]
        if(val == "yes"):
            initial_input[i] = 1
        elif(val == "no"):
            initial_input[i] = -1
    df = pd.read_csv("celeba_hq_dataset.csv")
    df = df.drop(df.index[len(df)-1])
    features = list(df)[1:]

    a = initial_input['Male']
    df = df[df['Male'] == a]
    count = len(df)
    while(len(df)>5):
        prev_df = len(df)
        pblts = {}
        total_rows = len(df)
        for a in features:
            if(a not in dont_ask):
                temp = Counter(df[a])
                pblts[a] = {-1 : round(temp[-1]/total_rows,4) , 1 : round(temp[1]/total_rows,2)} 

        get_new_df = {}
        for k,v in initial_input.items():
            temp = pblts[k]
            if(pblts[k] not in dont_ask): 
                get_new_df[k] = [v,temp[v]]

        min = 999
        for k,v in get_new_df.items():
            t1 = v
            if(t1[1] < min):
                min = t1[1]
                split_feature = k
                split_value = t1[0]
        print(split_feature,split_value)

        df = df[df[split_feature] == split_value]
        if(prev_df == len(df)):
            break
    x = list(df['index'])
    #x = dat_image(x)
    return render(request,"show_image_option.html" , {'initial_input': initial_input , 'df': x , 'count' : count})
