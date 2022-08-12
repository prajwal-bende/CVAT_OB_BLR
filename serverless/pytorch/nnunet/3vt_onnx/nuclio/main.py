import json
import base64
import io
from PIL import Image
import time
import os.path as osp
import numpy as np
import matplotlib
import nibabel as nb
import SimpleITK as sitk
import matplotlib.pyplot as plt
from pynvml import *
import os
import matplotlib
matplotlib.use('Agg')
import cv2
import subprocess
import onnx
import onnxruntime
import torch
import torchvision



from einops import rearrange,repeat,reduce

from skimage.measure import approximate_polygon, subdivide_polygon, find_contours

os.environ['RESULTS_FOLDER'] = 'nnUNet_trained_models'
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),

])
def get_gpu_lowest_usage():
  nvmlInit()
  gpu_free_mem = []
  for  i in range(nvmlDeviceGetCount()):
      handle = nvmlDeviceGetHandleByIndex(i)
      info = nvmlDeviceGetMemoryInfo(handle)

      gpu_free_mem.append(info.free/2**20)

  gpu_num = gpu_free_mem.index(max(gpu_free_mem))
  return gpu_num



def get_contour(gray_image,idx,flag13=False):
    if flag13:
        data=[2,3,4,5,6,7,9,11,12,13]
    else:
        data=mapping_dict[str(idx)]
    im1=gray_image.copy()
    stm=im1==data[0]
    for dt in data[1:]:
        stm1=im1==dt
        stm=stm | stm1

    im1[np.where(stm)]=255
    im1[np.where(im1!=255)]=0
    contours, _ = cv2.findContours(im1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = find_contours(np.flipud(np.rot90(im1)))
    #print(len(contours))
    c_area=np.array([cv2.contourArea(cv2.UMat(np.expand_dims(cnt.astype(np.float32), 1))) for cnt in contours])
    print(c_area)
    max_contour=contours[np.argmax(c_area)]
    #print(max_contour.shape)
    #max_contour=max(contours, key = cv2.contourArea)
    #print(max_contour.shape)
    #epsilon = 0.002*cv2.arcLength(max_contour,True)
    #approx = cv2.approxPolyDP(max_contour,epsilon,True)
    #approx = subdivide_polygon(np.squeeze(max_contour,1), degree=2, preserve_ends=True)
    #approx = approximate_polygon(np.squeeze(max_contour,1), tolerance=1)
    #approx = subdivide_polygon(max_contour, degree=2, preserve_ends=True)
    approx = approximate_polygon(max_contour, tolerance=1)
    approx = np.expand_dims(approx,1)
    print(approx.shape)
    return approx

#GPU = get_gpu_lowest_usage()


def clean_files(dir):
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))


ant=['Aortic Arch','Ductal Arch','SVC','Spine','Trachea']

mapping_dict={}
for i in range(1,6):
    mapping_dict[str(i)]=[i]

def init_context(context):
    context.logger.info("Init context...  0%")
    context.logger.info("Init context...100%")

def handler(context, event):
    context.logger.info("Run nnUNet model")
    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"]))

    input_image_numpy = np.asarray(Image.open(buf))


    if len(input_image_numpy.shape)<3:
      input_image_numpy = np.stack((input_image_numpy,)*3, axis=-1)

   
   

    ort_session = onnxruntime.InferenceSession(onnx_path)

    original_img = input_image_numpy.copy()
    img = repeat(original_img,'h w->h w')
    print("Debug purpose",img.shape)
    img_y = transform(img)
    img_y = img_y.squeeze(0)
    print("Debug purpose",img_y.shape)

    tic = time.time()

    zx = ort_session.run(['fin_output_int'], {'input': to_numpy(img_y)})
    output=zx[0]
    tac = time.time()
    FPS = 1/(tac-tic)
    print("Beast FPS = ",FPS)
    print("Time taken for inference is ",tac-tic)
    #change output to uint8
    gray_image = output.astype(np.uint8)
  

    results = []
    for i in range(1,6):
        if i in gray_image:
            points=np.squeeze(get_contour(gray_image,i),axis=1)
            if len(points)>=3:
                results.append({
                    "label": ant[i-1],
                    "points": points.ravel().tolist(),
                    "type": "polygon",
                })

    return context.Response(body=json.dumps(results), headers={},
        content_type='application/json', status_code=200)
