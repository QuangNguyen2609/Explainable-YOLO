import argparse
import multiprocessing as mp
import os

import cv2
import numpy as np
import torch
from skimage import io
from torch import nn
from random import uniform, randint
import pandas as pd
from tqdm import tqdm
import tqdm as tqdm_display
import cv2
import hydra
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
from saicinpainting.training.data.datasets import make_default_val_dataset
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.utils import register_debug_signal_handlers
from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.evaluation.refinement import refine_predict
import os
import time
import argparse
import numpy as np
from models.gradcam import YOLOV5GradCAM, YOLOV5GradCAMPlusPlus, GuidedBackpropReLUModel
from models.yolo_v5_object_detector import YOLOV5TorchObjectDetector
import cv2
from torch.utils.data._utils.collate import default_collate
from deep_utils import Box, split_extension

##================================= ARGS ==================================================##
parser = argparse.ArgumentParser()
parser.add_argument('--model-path', type=str, default="yolov5s.pt", help='Path to the model')
parser.add_argument('--img-path', type=str, default='images/', help='input image path')
parser.add_argument('--output-dir', type=str, default='outputs', help='output dir')
parser.add_argument('--img-size', type=int, default=960, help="input image size")
parser.add_argument('--mask-threshold', type=float, default=0.5, help="CAM mask threshold")
parser.add_argument('--topk', type=int, default=25, help="number of gbp values to choose")
parser.add_argument('--target-layer', type=str, default='model_23_cv3_act',
                    help='The layer hierarchical address to which gradcam will applied,'
                         ' the names should be separated by underline')
parser.add_argument('--method', type=str, default='gradcam', help='gradcam or gradcampp')
parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
parser.add_argument('--names', type=str, default=None,
                    help='The name of the classes. The default is set to None and is set to coco classes. Provide your custom names as follow: object1,object2,object3')

args = parser.parse_args()


##================================= LAMA INIT CONFIG ==================================================##
out_ext = '.png'
checkpoint_path = "lama/experiments/mtsd_aus/epoch599/models/best.ckpt"
indir = "/home/harry/yolov5-gradcam/test"
outdir = "/home/harry/yolov5-gradcam/test_out"
dataset_lama = {'kind': 'default', 'img_suffix': '.png', 'pad_out_to_modulo': 8}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_config_path = os.path.join("lama/experiments/mtsd_aus/epoch599", 'config.yaml')
with open(train_config_path, 'r') as f:
    train_config = OmegaConf.create(yaml.safe_load(f))

train_config.training_model.predict_only = True
train_config.visualizer.kind = 'noop'
model_lama = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
model_lama.freeze()
model_lama.to(device)

os.makedirs(outdir, exist_ok=True)
os.makedirs(indir, exist_ok=True)

##================================= Variables init ==================================================##

target = 27
grey_inpainting_asr = 0.0
asr = 0


##================================= AUX FUNC ==================================================##

def norm_image(image):
    """
    :param image: [H,W,C]
    :return:
    """
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)

def gen_cam(image, mask):
    """
    :param image: [H,W,C]
    :param mask: [H,W],0~1
    :return: tuple(cam,heatmap)
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1]  # gbr to rgb

    cam = heatmap + np.float32(image)
    heatmap_image =  heatmap*255 + np.float32(image) 
    heatmap_image = cv2.cvtColor(heatmap_image, cv2.COLOR_BGR2RGB)
    return norm_image(cam), heatmap_image, heatmap

def get_res_img(bbox, mask, res_img):
    mask = mask.squeeze(0).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).detach().cpu().numpy().astype(
        np.uint8)
    heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    n_heatmat = (Box.fill_outer_box(heatmap, bbox) / 255).astype(np.float32)
    res_img = res_img / 255
    res_img = cv2.add(res_img, n_heatmat)
    res_img = (res_img / res_img.max())
    return res_img, n_heatmat


def put_text_box(bbox, cls_name, res_img):
    x1, y1, x2, y2 = bbox
    # this is a bug in cv2. It does not put box on a converted image from torch unless it's buffered and read again!
    cv2.imwrite('temp.jpg', (res_img * 255).astype(np.uint8))
    res_img = cv2.imread('temp.jpg')
    res_img = Box.put_box(res_img, bbox)
    res_img = Box.put_text(res_img, cls_name, (x1, y1))
    return res_img


def concat_images(images):
    w, h = images[0].shape[:2]
    width = w
    height = h * len(images)
    base_img = np.zeros((width, height, 3), dtype=np.uint8)
    for i, img in enumerate(images):
        base_img[:, h * i:h * (i + 1), ...] = img
    return base_img


def predict(refine=False):
    dataset = make_default_val_dataset(indir, **dataset_lama)
    for img_i in range(len(dataset)):
        mask_fname = dataset.mask_filenames[img_i]
        cur_out_fname = os.path.join(
            outdir, 
            os.path.splitext(mask_fname[len(indir):])[0] + out_ext
        )
        os.makedirs(os.path.dirname(cur_out_fname), exist_ok=True)
        batch = default_collate([dataset[img_i]])
        if refine:
            assert 'unpad_to_size' in batch, "Unpadded size is required for the refinement"
            # image unpadding is taken care of in the refiner, so that output image
            # is same size as the input image
            cur_res = refine_predict(batch, model_lama, **predict_config.refiner)
            cur_res = cur_res[0].permute(1,2,0).detach().cpu().numpy()
        else:
            with torch.no_grad():
                batch = move_to_device(batch, device)
                batch['mask'] = (batch['mask'] > 0) * 1
                batch = model_lama(batch)                    
                cur_res = batch["inpainted"][0].permute(1, 2, 0).detach().cpu().numpy()
                unpad_to_size = batch.get('unpad_to_size', None)
                if unpad_to_size is not None:
                    orig_height, orig_width = unpad_to_size
                    cur_res = cur_res[:orig_height, :orig_width]

        cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
        cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{outdir}/single_mask.png", cur_res)
        del batch, cur_res, mask_fname, cur_out_fname
    return
##================================= MAIN ==================================================##

def main(img_path):
    device = args.device
    input_size = (args.img_size, args.img_size)
    img = cv2.imread(img_path)
    img = cv2.resize(img, (1280, 720))
    print('[INFO] Loading the model')
    model = YOLOV5TorchObjectDetector(args.model_path, device, img_size=input_size,
                                      names=None if args.names is None else args.names.strip().split(","))
    torch_img = model.preprocessing(img[..., ::-1])
    print(torch_img.shape)
    if args.method == 'gradcam':
        saliency_method = YOLOV5GradCAM(model=model, layer_name=args.target_layer, img_size=input_size)
    elif args.method == 'gradcampp':
        saliency_method = YOLOV5GradCAMPlusPlus(model=model, layer_name=args.target_layer, img_size=input_size)
    else:
        saliency_method = GuidedBackpropReLUModel(model=model, layer_name=args.target_layer, img_size=input_size)
        
    tic = time.time()
    masks, logits, [boxes, _, class_names, _] = saliency_method(torch_img)
    print("total time:", round(time.time() - tic, 4))
    result = torch_img.squeeze(0).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).detach().cpu().numpy()
    result = result[..., ::-1]  # convert to bgr
    images = [result]
    for i, mask in enumerate(masks):
        bbox, cls_name = boxes[0][i], class_names[0][i]
        x1, y1, x2, y2 = bbox
        mask = mask[0].squeeze(0).detach().cpu().numpy()
        res_img = result.copy()
        if args.method == 'gbp':
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            print("[INFO] MASK: ", mask.max(), mask.min())
            # print top 10 values in the mask
            # mask *= 10
            topk = np.argpartition(mask.flatten(), -30)[-30:]
            topk = np.unravel_index(topk, mask.shape)
            # make a small circle around the top 10 values
            mask_roi = np.zeros_like(mask)
            cv2.imwrite("gbp_mask.png", mask_roi*255)
            for i in range(len(topk[0])):
                y, x = topk[0][i], topk[1][i]
                mask_roi[y-5:y+5, x-5:x+5] = 1
            mask_roi = mask_roi[x1:x2, y1:y2]
            RoI = res_img[x1:x2, y1:y2]
            image_cam, heatmap, _ = gen_cam(RoI, mask_roi)
        else:
            mask_cond = mask >= args.mask_threshold
            mask = mask_cond.astype(int)
            RoI = res_img[x1:x2, y1:y2]
            mask_roi = mask[x1:x2, y1:y2]
        # add the mask to the image
        print(mask_roi.shape)
        overlay = RoI + mask_roi[..., None] * 255
        overlay = np.where(overlay == 255, 0, overlay)
        print(mask_roi.max())
        cv2.imwrite(f"test.png", overlay) # GBP mask
        cv2.imwrite(f"{indir}/single_mask.png", mask_roi*255) # GBP mask
        cv2.imwrite(f"{indir}/single.png", RoI)
        predict()
        out_file = os.path.join(outdir, "single_mask.png")
        clean_RoI = cv2.imread(out_file)
        res_img[x1:x2, y1:y2] = clean_RoI
    torch_img = model.preprocessing(res_img[..., ::-1])
    mask, logits, [boxes, _, class_names, _] = saliency_method(torch_img)
    print("[INFO] PREDICTION AFTER SANITIZATION: ", class_names)
    img_name = split_extension(os.path.split(img_path)[-1], suffix='-res')
    output_path = f'{args.output_dir}/{img_name}'
    os.makedirs(args.output_dir, exist_ok=True)
    print(f'[INFO] Saving the final image at {output_path}')
    cv2.imwrite(output_path, res_img)


def folder_main(folder_path):
    device = args.device
    input_size = (args.img_size, args.img_size)
    print('[INFO] Loading the model')
    model = YOLOV5TorchObjectDetector(args.model_path, device, img_size=input_size,
                                names=None if args.names is None else args.names.strip().split(","))
    model.confidence = 0.9
    print("[DEBUG] cuda memory after loading model ", torch.cuda.memory_allocated()/10**8)
    if args.method == 'gradcam':
        saliency_method = YOLOV5GradCAM(model=model, layer_name=args.target_layer, img_size=input_size)
    elif args.method == 'gradcampp':
        saliency_method = YOLOV5GradCAMPlusPlus(model=model, layer_name=args.target_layer, img_size=input_size)
    else:
        saliency_method = GuidedBackpropReLUModel(model=model, layer_name=args.target_layer, img_size=input_size)
        topk = args.topk
        
    for item in os.listdir(folder_path):
        img_path = os.path.join(folder_path, item)
        img_name = split_extension(os.path.split(img_path)[-1], suffix='-res')
        output_path = f'{args.output_dir}/{img_name}'
        if os.path.exists(output_path):
            print("[INFO] skipping ", item)
            continue
        img_path = os.path.join(folder_path, item)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (1280, 720))
        print("[INFO] image shape: ",img.shape)
        torch_img = model.preprocessing(img[..., ::-1])
        tic = time.time()
        print("[DEBUG] cuda memory before mask ", torch.cuda.memory_allocated()/10**8)
        masks, logits, [boxes, _, class_names, _] = saliency_method(torch_img)
        print("[DEBUG] cuda memory after mask", torch.cuda.memory_allocated()/10**8)
        print("total time:", round(time.time() - tic, 4))
        result = torch_img.squeeze(0).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).detach().cpu().numpy()
        result = result[..., ::-1]  # convert to bgr
        if len(masks) == 0:
            print("[INFO] No object detected")
            continue
        for i, mask in enumerate(masks):
            bbox, cls_name = boxes[0][i], class_names[0][i]
            x1, y1, x2, y2 = bbox
            mask = mask[0].squeeze(0).detach().cpu().numpy()
            res_img = result.copy()
            if args.method == 'gbp':
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                mask *= 10
                top_points = np.argpartition(mask.flatten(), -topk)[-topk:]
                top_points = np.unravel_index(top_points, mask.shape)
                mask_roi = np.zeros_like(mask)
                cv2.imwrite("gbp_mask.png", mask_roi*255)
                for i in range(len(top_points[0])):
                    y, x = top_points[0][i], top_points[1][i]
                    mask_roi[y-10:y+10, x-10:x+10] = 1
                mask_roi = mask_roi[x1:x2, y1:y2]
                RoI = res_img[x1:x2, y1:y2]
            else:
                mask_cond = mask >= args.mask_threshold
                mask = mask_cond.astype(int)
                RoI = res_img[x1:x2, y1:y2]
                mask_roi = mask[x1:x2, y1:y2]
            cv2.imwrite(f"{indir}/single_mask.png", mask_roi*255) # GBP mask
            cv2.imwrite(f"{indir}/single.png", RoI)
            predict()
            out_file = os.path.join(outdir, "single_mask.png")
            clean_RoI = cv2.imread(out_file)
            res_img[x1:x2, y1:y2] = clean_RoI
        img_name = split_extension(os.path.split(img_path)[-1], suffix='-res')
        output_path = f'{args.output_dir}/{img_name}'
        os.makedirs(args.output_dir, exist_ok=True)
        # print(f'[INFO] Saving the final image at {output_path}')
        cv2.imwrite(output_path, res_img)
        # del torch_img, masks, logits, boxes, class_names, result, img, mask, mask_cond, mask_roi, RoI, clean_RoI, model, saliency_method
        # print("[DEBUG] cuda memory after del ", torch.cuda.memory_allocated()/10**8)

if __name__ == '__main__':
    if os.path.isdir(args.img_path):
        folder_main(args.img_path)
    else:
        main(args.img_path)


