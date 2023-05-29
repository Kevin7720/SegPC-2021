import numpy as np 
from skimage.io import imread, imshow, imsave
import glob
import skimage.morphology as sm

# im_files = glob.glob('/home/kevin/Mask_RCNN/2020nucleus/dataset/stage1_train/*')
# for file in im_files:
#     name = file.split('/')[-1]
#     im = imread('/home/kevin/Mask_RCNN/2020nucleus/dataset/all_pre/' + name + '.png')
#     mask1 = glob.glob(file + '/masks_1/*')
#     mask2 = glob.glob(file + '/masks_2/*')
#     for one in mask1:
#         m1 = imread(one)
#         dst1=sm.dilation(m1,sm.square(3)) - sm.erosion(m1,sm.square(3))
#         im[:,:,0][np.where(dst1==255)] = 255
#         im[:,:,1][np.where(dst1==255)] = 0
#         im[:,:,2][np.where(dst1==255)] = 0
#     for two in mask2:
#         m2 = imread(two)
#         dst2=sm.dilation(m2,sm.square(3)) - sm.erosion(m2,sm.square(3))
#         im[:,:,2][np.where(dst2==255)] = 0
#         im[:,:,1][np.where(dst2==255)] = 255
#         im[:,:,0][np.where(dst2==255)] = 0
#     imsave('/home/kevin/Mask_RCNN/2020nucleus/dataset/new_label2/' + name + '.png',im)


def label_to_data(name,stage='A'):
    new_path  = '/home/kevin/Mask_RCNN/2020nucleus/dataset/new_label/' + str(name) +'_json'
    save_path = '/home/kevin/Mask_RCNN/2020nucleus/dataset/stage1_train/'
    label = imread(new_path + '/label.png')[:,:,0]
    n = 0
    for i in range(1,label.max()+1):
        n = n + 1
        new = np.zeros(label.shape)
        new[np.where(label == i)] = 255
        imsave(save_path + str(name) + '/' + stage + '_' + str(n) + '.bmp', new)
def reset_data(name):
    path = '/home/kevin/Mask_RCNN/2020nucleus/dataset/stage1_train' + '/' + str(name)
    im = imread(path + '/images/' + str(name) + '.bmp')
    for one in glob.glob(path + '/masks_1/*.bmp'):
        m1 = imread(one)
        dst1=sm.dilation(m1,sm.square(3)) - sm.erosion(m1,sm.square(3))
        im[:,:,0][np.where(dst1==255)] = 255
        im[:,:,1][np.where(dst1==255)] = 0
        im[:,:,2][np.where(dst1==255)] = 0
    for two in glob.glob(path + '/masks_2/*.bmp'):
        m2 = imread(two)
        dst2=sm.dilation(m2,sm.square(3)) - sm.erosion(m2,sm.square(3))
        im[:,:,2][np.where(dst2==255)] = 0
        im[:,:,1][np.where(dst2==255)] = 255
        im[:,:,0][np.where(dst2==255)] = 0
    imsave('/home/kevin/Mask_RCNN/2020nucleus/dataset/new_label/' + str(name) + '.png',im)
import os
import shutil
from IPython import get_ipython
name = 303
if os.path.isfile('/home/kevin/Mask_RCNN/2020nucleus/dataset/new_label/' + str(name) + '.json'):
    path__ = 'sx labelme_json_to_dataset /home/kevin/Mask_RCNN/2020nucleus/dataset/new_label/' + str(name) + '.json'
    get_ipython().magic(path__)
    label_to_data(name,'A')
    os.remove('/home/kevin/Mask_RCNN/2020nucleus/dataset/new_label/'+ str(name) + '.json')
    shutil.rmtree('/home/kevin/Mask_RCNN/2020nucleus/dataset/new_label/' + str(name) + '_json')
else:
    reset_data(name)

