import json
import base64
import io
from PIL import Image

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
from skimage.measure import approximate_polygon, subdivide_polygon, find_contours

os.environ['RESULTS_FOLDER'] = 'nnUNet_trained_models'

def get_gpu_lowest_usage():
  nvmlInit()
  gpu_free_mem = []
  for  i in range(nvmlDeviceGetCount()):
      handle = nvmlDeviceGetHandleByIndex(i)
      info = nvmlDeviceGetMemoryInfo(handle)

      gpu_free_mem.append(info.free/2**20)

  gpu_num = gpu_free_mem.index(max(gpu_free_mem))
  return gpu_num

def convert_2d_image_to_nifti(input_filename: str, output_filename_truncated: str,is_seg: bool, spacing=(999, 1, 1),
                              transform=None, ) -> None:
    
    img = Image.open(input_filename)
    img = img.convert('L')
    img = np.asarray(img)
    if transform is not None:
        img = transform(img)

    if len(img.shape) == 2:  # 2d image with no color channels
        img = img[None, None]  # add dimensions
    else:
        assert len(img.shape) == 3, "image should be 3d with color channel last but has shape %s" % str(img.shape)
        # we assume that the color channel is the last dimension. Transpose it to be in first
        img = img.transpose((2, 0, 1))
        # add third dimension
        img = img[:, None]

    # image is now (c, x, x, z) where x=1 since it's 2d
    if is_seg:
        assert img.shape[0] == 1, 'segmentations can only have one color channel, not sure what happened here '

    for j, i in enumerate(img):

        if is_seg:
            i = i.astype(np.uint32)

        itk_img = sitk.GetImageFromArray(i)
        itk_img.SetSpacing(list(spacing)[::-1])
        
        
        sitk.WriteImage(itk_img, output_filename_truncated + "_%04.0d.nii.gz" % j)


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

GPU = get_gpu_lowest_usage()
task=804
FOLD=5
uploaded_image_folder='input_2d_png'
filename='4CH.png'
uploaded_image_folder_nii='input_nii'
uploaded_image_folder_nii_output='output_nii'
output_image_folder='output_2d_png'

def clean_files(dir):
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))

input_image_path = osp.join(uploaded_image_folder,filename)
output_image_path= osp.join(output_image_folder,filename)
ant=['Outer Thorax','Heart Contour','IV Septum','RA','LA','RV','LV','dAorta','Atrium Septum','Spine Triangle','Mitral Valve','Tricuspid Valve','AV Septum','PV1 (RPV)','PV2 (LPV)']

mapping_dict={'1':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
             '2':[2,3,4,5,6,7,9,11,12,13,14,15]}
for i in range(3,16):
    mapping_dict[str(i)]=[i]

def init_context(context):
    context.logger.info("Init context...  0%")
    context.logger.info("Init context...100%")

def handler(context, event):
    context.logger.info("Run nnUNet model")
    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"]))

    input_image_numpy = np.asarray(Image.open(buf))

    clean_files(uploaded_image_folder)
    clean_files(uploaded_image_folder_nii)
    clean_files(uploaded_image_folder_nii_output)
    clean_files(output_image_folder)

    if len(input_image_numpy.shape)<3:
      input_image_numpy_ei = np.stack((input_image_numpy,)*3, axis=-1)
      matplotlib.image.imsave(input_image_path, input_image_numpy_ei)
    else:
      matplotlib.image.imsave(input_image_path, input_image_numpy)

    nii_converted_path = osp.join(uploaded_image_folder_nii,filename[:-4])
    convert_2d_image_to_nifti(input_image_path, nii_converted_path, is_seg=False)
    os.system(f"CUDA_VISIBLE_DEVICES={GPU} nnUNet_predict -i {uploaded_image_folder_nii} -o {uploaded_image_folder_nii_output} -t {task} -m 2d -f {FOLD}")
    #os.system("ls -l")
    #os.system("nnUNet_predict")
    #command= "nnUNet_predict -i " + uploaded_image_folder_nii + " -o " + uploaded_image_folder_nii_output + " -t " + str(task) + " -m 2d -f " + str(FOLD)
    #print(command)
    #list_files = subprocess.run(command.split(),check=True, text=True)
    #print("The exit code was: %d" % list_files.returncode)
    val_image = np.flip(np.rot90(np.array(nb.load(osp.join(uploaded_image_folder_nii_output,filename[:-4]+'.nii.gz')).dataobj),3),axis=1)
    cv2.imwrite(output_image_path, np.squeeze(val_image))

    gray_image=np.squeeze(val_image)
    print(gray_image.shape)
    print(np.unique(gray_image))
    results = []
    for i in range(1,16):
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
