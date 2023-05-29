import numpy as np
from skimage.io import imread, imshow, imsave
from skimage.transform import resize
import glob
from mrcnn.config import Config
from mrcnn import model as modellib
import pdb
import skimage.morphology as sm
import os 
path = '/media/xuan/Transcend/cell/val_x'
image_pathes = sorted(glob.glob(path + '/*.bmp'))

class NucleusConfig(Config):
    NAME = "nucleus"
    IMAGES_PER_GPU = 2
    NUM_CLASSES = 1 + 2  # Background + nucleus
    DETECTION_MIN_CONFIDENCE = 0.34
    BACKBONE = "resnet101"    
    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_MIN_SCALE = 2.0
    RPN_ANCHOR_SCALES = (64,80,96,112,128)#(AA + B,AA + B*2,AA + B*3,AA + B*4,AA + B*5)#(64,80,96,112,128)
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]
    RPN_ANCHOR_STRIDE = 2
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]
    POST_NMS_ROIS_TRAINING = 2000
    POST_NMS_ROIS_INFERENCE = 3000
    RPN_NMS_THRESHOLD = 0.9
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256  #datagen生成gt的數量\
    MEAN_PIXEL = np.array([205.95, 172.00, 184.157])
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56,56)  # (height, width) of the mini-mask\
    TRAIN_ROIS_PER_IMAGE = 200
    MAX_GT_INSTANCES = 50
    DETECTION_MAX_INSTANCES = 100
    DETECTION_NMS_THRESHOLD = 0.7
    ROI_POSITIVE_RATIO = 0.33
    Data_Set_Nms_max =  0.7
    Data_Set_Nms_min =  0.3
class NucleusInferenceConfig(NucleusConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_RESIZE_MODE = "pad64"
    RPN_NMS_THRESHOLD = 0.9
    
def model_cell_load(weight_path):
    config = NucleusInferenceConfig()
    model  = modellib.MaskRCNN(mode="inference", config=config,model_dir=weight_path)
    model.load_weights(weight_path, by_name=True)
    return model
def iou(l1,l2):
    smooth = 0.0000001
    inter = (l1*l2).sum()
    all_  = l1.sum() + l2.sum()
    return (inter + smooth)/ (all_ + smooth)
model_weight = glob.glob('./*.h5')
model = model_cell_load(model_weight[0])
for image_path in image_pathes:
    image = imread(image_path).astype(np.float)
    name  = image_path.split('/')[-1].split('.')[0]
    ori_shape  = image.shape 
    shape_size = max(np.ceil(image.shape[0]/512),np.ceil(image.shape[1]/512))
    shape_size = (np.int(image.shape[0]/shape_size),np.int(image.shape[1]/shape_size))
    image = resize(image,shape_size)
    no_pad_shape = image.shape
    pad = np.ones([512,512,3])
    pad[:,:,0] = pad[:,:,0]*163
    pad[:,:,1] = pad[:,:,1]*151
    pad[:,:,2] = pad[:,:,2]*178
    pad[0:image.shape[0],0:image.shape[1],:] = image
    r       = model.detect([pad], verbose=0)[0]
    save_number = 1
    while True:
        # pdb.set_trace()
        if len(np.where(r['class_ids'] == 1)[0]) > 0:
            aa = r['class_ids'].copy()
            number = np.where(r['class_ids'] == 1)[0][0]
            r['class_ids'][number] = -1
            l1 = r['masks'][:,:,number]
            l2_marx = np.ones(r['class_ids'].shape)*-1
            for l2_number in np.where(r['class_ids'] == 2)[0].tolist():
                l2 = r['masks'][:,:,l2_number]
                l2_marx[l2_number] = iou(l1,l2)
            if l2_marx.max() > 0.00001 :
                l2 = r['masks'][:,:,np.where(l2_marx == l2_marx.max())[0][0]]
                r['class_ids'][np.where(l2_marx == l2_marx.max())[0][0]] = -1
                l2 = l2[0:no_pad_shape[0],0:no_pad_shape[1]]
                l2 = np.where(resize(l2,ori_shape[0:2])>0,1,0).astype(np.float) * 20
                l1 = l1[0:no_pad_shape[0],0:no_pad_shape[1]]
                l1 = np.where(resize(l1,ori_shape[0:2])>0,1,0).astype(np.float) * 20
                
                imsave('./data_pred/' + name + '_' + str(save_number) + '.bmp',(l1+l2).astype(np.uint8))
                save_number = save_number + 1
                l1, l2 = None, None 
            else:
                l1 = l1[0:no_pad_shape[0],0:no_pad_shape[1]]
                l1 = np.where(resize(l1,ori_shape[0:2])>0,1,0).astype(np.float) * 20
                imsave('./data_pred/' + name + '_' + str(save_number) + '.bmp',l1.astype(np.uint8))
                save_number = save_number + 1
                l1,l2 = None,None
                print('Did not find label2 save in ' + './' + name + '_' + str(save_number))
        else:
            l1,l2 = None,None
            break
    if len(np.where(r['class_ids'] == 2)[0]) != 0:
        # pdb.set_trace()
        for nn in np.where(r['class_ids'] == 2)[0].tolist():
            l2 = r['masks'][:,:,nn]
            l2 = l2[0:no_pad_shape[0],0:no_pad_shape[1]]
            l2 = np.where(resize(l2,ori_shape[0:2])>0,1,0).astype(np.float)
            l1 = np.where(sm.dilation(l2,sm.square(5))>0,1,0).astype(np.float)
            imsave('./data_pred/' + name + '_' + str(save_number) + '.bmp',(l1*20+l2*20).astype(np.uint8))
            r['class_ids'][nn] = -1
            l1,l2 = None,None
            print('Did not find label1 save in ' + './data_pred/' + name + '_' + str(save_number))
            
            
        
        
        
        
    
