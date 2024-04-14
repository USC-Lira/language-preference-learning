import os
import torch
import argparse
import multiprocessing as mp
import numpy as np
import tqdm
import time
import sys
import cv2

from einops import rearrange

sys.path.insert(0, f'Detic/')
sys.path.insert(0, f'Detic/third_party/CenterNet2/')

from centernet.config import add_centernet_config

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from detic.predictor import VisualizationDemo

from detic.config import add_detic_config

from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide

# constants
WINDOW_NAME = "Detic"


def setup_cfg(args):
    cfg = get_cfg()
    if args.cpu:
        cfg.MODEL.DEVICE = "cpu"
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'  # load later
    if not args.pred_all_class:
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", help="Take inputs from webcam.")
    parser.add_argument("--cpu", action='store_true', help="Use CPU only.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
             "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
             "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--vocabulary",
        default="lvis",
        choices=['lvis', 'openimages', 'objects365', 'coco', 'custom'],
        help="",
    )
    parser.add_argument(
        "--custom_vocabulary",
        default="",
        help="",
    )
    parser.add_argument("--pred_all_class", action='store_true')
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--data-dir",
        default="data/data_img_obs_res_224_30k/train",
        help="The directory containing the data.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="The start index of the data.",
    )
    return parser



def detect_objects_in_image(imgs, predictor):
    """
    Detect objects in an image using a predictor.

    Parameters:
        - imgs: The images of shape (batch_size, height, width, channels). 
        - predictor: The predictor.

    Returns:
        - pred_boxes: The predicted boxes.
    """
    height, width = imgs[0].shape[:2]

    inputs = []
    for img in imgs:
        if img.max() <= 1:
            img = img * 255.0
        img = img.astype(np.uint8)
        img = predictor.aug.get_transform(img).apply_image(img)
        img = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))
        img.to(predictor.cfg.MODEL.DEVICE)
        inputs.append({"image": img, "height": height, "width": width})

    predictions = predictor.model(inputs)
    
    # return prediction boxes
    pred_boxes = [pred["instances"].pred_boxes.tensor.detach() for pred in predictions]
    return pred_boxes


def prepare_image(image, transform, device):
    image = transform.apply_image(image)
    image = torch.as_tensor(image, device=device) 
    return image.permute(2, 0, 1).contiguous()


def segment_objects(imgs, pred_boxes, model):
    """
    Segment objects in an image using SAM and the predicted boxes.

    Parameters:
        - imgs: The images of shape (batch_size, height, width, channels).
        - pred_boxes: The predicted boxes.
        - model: The SAM model.

    Returns:
        - seg_masks: The segmented masks.
    """
    batched_input = []
    resize_transform = ResizeLongestSide(model.image_encoder.img_size)
    if imgs[0].max() <= 1:
        imgs = imgs * 255.0
        imgs = imgs.astype(np.uint8)

    for img, boxes in zip(imgs, pred_boxes):
        input = {
            "image": prepare_image(img, resize_transform, model.device),
            "boxes": resize_transform.apply_boxes_torch(boxes, img.shape[:2]),
            "original_size": img.shape[:2],
        }
        batched_input.append(input)
    
    batched_output = model(batched_input, multimask_output=False)
    seg_masks = [output["masks"].detach().cpu().numpy() for output in batched_output]

    return seg_masks



def apply_segmentation_mask(img, masks):
    """
    Apply a segmentation mask to an image.

    Parameters:
        - img: The image.
        - mask: The mask.
        - color: The color of the mask.
        - alpha: The alpha value of the mask.
    
    Returns:
        - The image with the mask applied.
    """
    img = img.copy()
    final_mask = np.zeros((img.shape[0], img.shape[1]))
    for mask in masks:
        final_mask = np.maximum(final_mask, mask)

    final_mask = np.stack([final_mask] * 3, axis=-1).squeeze()
    img = final_mask * img
    return img



if __name__ == "__main__":
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg, args)
    predictor = demo.predictor

    data_dir = args.data_dir
    img_obs = np.load(f'{data_dir}/traj_img_obs.npy')

    sam_checkpoint = "sam_vit_b_01ec64.pth"
    model_type = "vit_b"
    device = torch.device("cuda")

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    for i in tqdm.tqdm(range(args.start_index, len(img_obs))):
        traj_img_obs = img_obs[i]
        traj_seg_img_obs = np.zeros_like(traj_img_obs)

        # Process images in the trajectory by batch
        batch_size = 4
        num_batches = len(traj_img_obs) // batch_size

        for j in tqdm.tqdm(range(num_batches), leave=False):
            curr_img_obs = traj_img_obs[j * batch_size: (j + 1) * batch_size]

            # Detect objects in the images
            try:
                pred_boxes = detect_objects_in_image(curr_img_obs, predictor)
            except RuntimeError as e:
                print(e)
                print(f'Error in traj {i}, batch {j}')
                break

            # Segment objects in the images
            try:
                seg_masks = segment_objects(curr_img_obs, pred_boxes, sam)
            except RuntimeError as e:
                print(e)
                print(f'Error in traj {i}, batch {j}')
                break

            seg_imgs = []
            for k in range(len(curr_img_obs)):
                img = curr_img_obs[k]
                mask = seg_masks[k]

                # Apply the mask to the image
                seg_img = apply_segmentation_mask(img, mask)
                seg_imgs.append(seg_img)

                # Save the image
                seg_img = seg_img * 255.0
                seg_img = seg_img.astype(np.uint8)
                seg_img = cv2.cvtColor(seg_img, cv2.COLOR_RGB2BGR)
                if j == 0 and k == 0:
                    save_dir = f'{data_dir}/seg_img_obs_example'
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    cv2.imwrite(f'{save_dir}/{i}_{j}_{k}.png', seg_img)
            
            seg_imgs = rearrange(seg_imgs, 'b h w c -> b h w c')
            traj_seg_img_obs[j * batch_size: (j + 1) * batch_size] = seg_imgs
        
        seg_data_dir = f'{data_dir}/seg_img_obs'
        if not os.path.exists(seg_data_dir):
            os.makedirs(seg_data_dir)
        np.save(f'{seg_data_dir}/{i}.npy', traj_seg_img_obs)
