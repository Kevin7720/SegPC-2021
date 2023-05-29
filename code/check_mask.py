import numpy as np
from skimage.io import imread, imshow, imsave
import glob
import skimage
import cv2

file_name = '/media/xuan/Transcend/cell/data/ori_data/stage1_train_1024'
save_path = '/media/xuan/Transcend/cell/data/newlabel'


im_path = sorted(glob.glob(file_name + '/*'))
kernel = np.ones((3,3))
for im_number in im_path:
    image_path  = glob.glob(im_number + '/images/*')
    mask_b_path = glob.glob(im_number + '/masks_1/*')
    mask_s_path = glob.glob(im_number + '/masks_2/*')
    im = imread(image_path[0]).astype(np.float)
    for mask_b in mask_b_path:
        b = imread(mask_b)
        b = b - cv2.erode(b,kernel)
        im[:,:,0][np.where(b ==255 )] = 255
        im[:,:,1][np.where(b ==255 )] = 0
        im[:,:,2][np.where(b ==255 )] = 0
    for mask_s in mask_s_path:
        s = imread(mask_s)
        s = s - cv2.erode(s,kernel)
        im[:,:,0][np.where(s ==255 )] = 0 
        im[:,:,1][np.where(s ==255 )] = 255
        im[:,:,2][np.where(s ==255 )] = 0
    name = im_number.split('/')[-1]
    imsave(save_path + '/' + name + '.jpg',im)
