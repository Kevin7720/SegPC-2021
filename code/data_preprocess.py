from skimage.io import imread, imshow, imsave
import glob
import os
import shutil
import numpy as np
from skimage.transform import resize
import pdb
import imageio.core.util
from skimage import measure

def ignore_warnings(*args, **kwargs):
    pass
imageio.core.util._precision_warn = ignore_warnings

X = glob.glob('/home/xuan/Downloads/train/x/*.bmp')
for im in X:
    name    = im.split('/')[-1].split('.')[0]
    gt_file = glob.glob(im.split('x/')[0] + 'y/' + name + '_*.bmp')
    image = imread(im).astype(np.float)
    image_o = image
    shape_size = max(np.ceil(image.shape[0]/512),np.ceil(image.shape[1]/512))
    shape_size = (np.int(image.shape[0]/shape_size),np.int(image.shape[1]/shape_size))
    image = resize(image,shape_size)
    pad = np.ones([512,512,3])
    pad[:,:,0] = pad[:,:,0]*163
    pad[:,:,1] = pad[:,:,1]*151
    pad[:,:,2] = pad[:,:,2]*178
    pad[0:image.shape[0],0:image.shape[1],:] = image
    if not os.path.isdir('/home/xuan/Downloads/train/stage1_train/' + name):
        os.mkdir('/home/xuan/Downloads/train/stage1_train/' + name)
        os.mkdir('/home/xuan/Downloads/train/stage1_train/' + name + '/images')
        os.mkdir('/home/xuan/Downloads/train/stage1_train/' + name + '/masks_1')
        os.mkdir('/home/xuan/Downloads/train/stage1_train/' + name + '/masks_2')
    imsave('/home/xuan/Downloads/train/stage1_train/' + name + '/images/' + name + '.bmp',pad)
    number = 0
    m3_number = 0
    for gt in gt_file:
        gt_im  = imread(gt)
        if gt_im.shape[-1] == 3:
            gt_im = gt_im[:,:,0]
        assert gt_im.shape == image_o.shape[0:2]
        gt_im = resize(gt_im,shape_size)
        pad_gt = np.zeros([512,512])
        pad_gt[0:gt_im.shape[0],0:gt_im.shape[1]] = gt_im
        gt_im = pad_gt*255
        gt_im1 = np.zeros(gt_im.shape)
        gt_im2 = np.zeros(gt_im.shape)
        gt_im1[np.where(gt_im != 0)] = 255
        gt_im2[np.where(gt_im >= 35)] = 255
        label_image = measure.label(gt_im2)   
        if (gt_im1/255).sum()*0.7 > (gt_im2/255).sum():
            imsave('/home/xuan/Downloads/train/stage1_train/' + name + '/masks_1/' + str(number) + '.bmp',gt_im1)
            imsave('/home/xuan/Downloads/train/stage1_train/' + name + '/masks_2/' + str(number) + '.bmp',gt_im2)
        else :
            m3_number = m3_number + 1
            imsave('/home/xuan/Downloads/train/stage1_train/' + name + '/masks_1/' + str(number) + '.bmp',gt_im1)
            
        number = number + 1
    if m3_number != 0:
        print(name + 'has mask 3 : ' + str(m3_number))
# import cv2

# img = cv2.imread('/home/xuan/Downloads/train/y/106_1.bmp')


# def click_info(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDBLCLK:
#         print('pos', x, y)
#         b, g, r = img[y, x]
#         print("b g r", b, g, r)

# cv2.namedWindow('image')
# cv2.setMouseCallback('image',click_info)

# while True:
#     cv2.imshow("image", img)
#     if cv2.waitKey(20) & 0xFF ==27:
#         break

# cv2.destroyAllWindows()
    