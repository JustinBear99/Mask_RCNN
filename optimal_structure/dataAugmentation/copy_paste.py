import os
import cv2
import json
import tqdm
import copy
import random
import base64
import argparse
import datetime
import numpy as np
from PIL import Image, ImageDraw
from skimage import measure

def mask2box(mask):
    """Convert binary mask to bounding box(x, y, w, h)

    Args:
        mask (ndarray): binary mask

    Returns:
        list: bbox(x, y, w, h)
    """
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    # print(mask.shape)
    a = np.where(mask != 0)
    # print(a)
    y1, y2, x1, x2 = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
    return [x1, y1, x2-x1, y2-y1]

def points2box(points):
	xmin, ymin, xmax, ymax = points[0][0], points[0][1], points[0][0], points[0][1]
	for point in points[1:]:
		if point[0] > xmax:
			xmax = point[0]
		if point[1] > ymax:
			ymax = point[1]
		if point[0] < xmin:
			xmin = point[0]
		if point[1] < ymin:
			ymin = point[1]
	return [int(xmin), int(ymin), int(xmax-xmin), int(ymax-ymin)]

def annos2masks(annos, width, height):
    masks = []
    for it in annos:
        if it['label'] == 'spear':
            img = Image.new('L', (width, height), 0)
            points_tuple = [ (point[0], point[1]) for point in it['points'] ]
            ImageDraw.Draw(img).polygon(points_tuple, outline=1, fill=1)
            mask = np.array(img)
        else:
            mask = it['points']
        masks.append({'mask': mask, 'bbox': points2box(it['points']), 'label': it['label'], 'shape_type': it['shape_type']})
    return masks

def masks2annos(masks):
    annos = []
    for mask in masks:
        if mask['label'] == 'spear':
            segmentation = measure.find_contours(mask["mask"].T, 0.5)
            # print(segmentation)
            for seg in segmentation:
                # print(seg)
                new_seg = []
                if len(seg) > 30:
                    for i in range(0, len(seg), int(len(seg)/30)):
                        new_seg.append(seg[i].tolist())
                else:
                	new_seg = [ s.tolist() for s in seg]
                annos.append(dict(
                    label=mask['label'], points=new_seg, group_id=None, shape_type=mask['shape_type'], flags=dict()
                ))
        else:
            annos.append(dict(
                label=mask['label'], points=mask['mask'], group_id=None, shape_type=mask['shape_type'], flags=dict()
                ))
    return annos

def readAnnos(image_list, annotations, image_name=None, spear_only=False):
    """Extract the annotations of specific or random image from annotations.json

    Args:
        image_list (list): List of image names.
        annotations (dict): Content of annotations.json
        image_name (str, optional): Specific image or None for random. Defaults to None.

    Returns:
        annos (list): List of annotations(dicts).
        width (int): Width of the image
        height (int): height of the image
    """
    image_name = random.shuffle(image_list)[0] if image_name == None else image_name
    for it in annotations['images']:
        if it['file_name'] == 'JPEGImages/' + image_name:
            image_id = it['id']
            height = it["height"]
            width = it["width"]
    annos = []
    for it in annotations['annotations']:
        if spear_only:
            if it['image_id'] == image_id and it['category_id'] == 3:
                annos.append(it)
        else:
            if it['image_id'] == image_id:
                annos.append(it)
    return annos, width, height

def LSJ(img, masks, min_scale=0.1, max_scale=2.0):
    rescale_ratio = np.random.uniform(min_scale, max_scale)
    h, w, c = img.shape

    # rescale
    h_new, w_new = int(h * rescale_ratio), int(w * rescale_ratio)
    img = np.resize(img, (h_new, w_new, c))
    for mask in masks:
        mask['mask'] = np.resize(mask['mask'], (h_new, w_new))

    x, y = int(np.random.uniform(0, abs(w_new - w))), int(np.random.uniform(0, abs(h_new - h)))
    if rescale_ratio <= 1.0:  # padding
        img_pad = np.ones((h, w, 3), dtype=np.uint8) * 168
        img_pad[y:y+h_new, x:x+w_new, :] = img
        masks_pad = []
        for mask in masks:
            mask_pad = np.zeros((h, w), dtype=np.uint8)
            mask_pad[y:y+h_new, x:x+w_new] = mask['mask']
            masks_pad.append({'mask':mask_pad, 'label': mask['label'], 'shape_type': mask['shape_type']})
        return img_pad, masks_pad
    else:  # crop
        img_crop = img[y:y+h, x:x+w, :]
        masks_crop = []
        for mask in masks:
            masks_crop.append({'mask': mask['mask'][y:y+h, x:x+w], 'label': mask['label'], 'shape_type': mask['shape_type']})
        return img_crop, masks_crop

def img_add(image, masks, image_2, masks_2):
    """Cut all the instance from image2 to random paste on image

    Args:
        image (3darray): rgb image
        masks (list): list of masks of image
        image_2 (3darray): image used for cut and paste
        masks_2 (list): list of masks of image_2

    Returns:
        image (3darray): pasted image
        masks (list): pasted masks
    """
    height, width, channel = image.shape
    # print(height, width)
    for mask_2 in masks_2:
        if mask_2['label'] == 'spear':
            # [x, y, w, h] = mask2box(mask_2['mask'])
            [x, y, w, h] = mask_2['bbox']
            image_2_sub = image_2 * np.stack((mask_2['mask'],)*3, axis=-1)
            image_2_sub = image_2_sub[y:y+h, x:x+w, :]
            mask_2_sub = mask_2['mask'][int(y):int(y+h), int(x):int(x+w)]

            new_x, new_y = random.randint(0, width-w), random.randint(0, height-h)

            new_image_2 = np.zeros((height, width, 3), dtype=np.uint8)
            new_image_2[new_y:new_y+h, new_x:new_x+w, :] = image_2_sub
            new_mask_2 = np.zeros((height, width), dtype=np.uint8)
            new_mask_2[new_y:new_y+h, new_x:new_x+w] = mask_2_sub
            new_image_sub = image * np.stack((new_mask_2,)*3, axis=-1)
            image -= new_image_sub
            image += new_image_2
            # [x, y, w, h] = mask2box(mask_2['mask'])
            # print(x, y, w, h)
            # image_2_sub = image_2 * np.stack((mask_2['mask'],)*3, axis=-1)
            # image_2_sub = image_2_sub[int(y):int(y+h), int(x):int(x+w), :]
            # mask_2_sub = mask_2['mask'][int(y):int(y+h), int(x):int(x+w)]
            # print(width-w, height-h)
            # new_x, new_y = random.randint(0, width-w), random.randint(0, height-h)
            # roi = image[new_y:h, new_x:w]
            # mask_inv = cv2.bitwise_not(mask_2_sub)
            # image_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
            # image_2_fg = cv2.bitwise_and(image_2_sub, image_2_sub, mask=mask_2_sub)
            # dst = cv2.add(image_bg, image_2_fg)
            # image[new_y:h, new_x:w] = dst
            # new_mask_2 = np.zeros((height, width), dtype=np.uint8)
            # new_mask_2[new_y:h, new_x:w] = mask_2_sub
            masks.append({'mask': new_mask_2, 'label': mask_2['label'], 'shape_type': mask_2['shape_type']})
    return image, masks

def copy_paste(image, masks, image_2, masks_2):
    """Perform copy-paste work flow

    Args:
        image (3darray): image to be pasted
        masks (list): [description]
        image_2 ([type]): [description]
        masks_2 ([type]): [description]

    Returns:
        [type]: [description]
    """
    # image, masks = LSJ(image, masks)
    # image_2, masks_2 = LSJ(image_2, masks_2)

    image, masks = img_add(image, masks, image_2, masks_2)
    return image, masks

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="/home/ubuntu/detectron2/datasets/coco/annotations/train", help="Input image directory.")
    parser.add_argument("--input_dir_2")
    parser.add_argument("--output_dir", default="/home/ubuntu/detectron2/copypaste")
    args = parser.parse_args()
    return args

def main(args):
    # read input images and annotations
    image_list = [ image for image in os.listdir(args.input_dir) if image.endswith('.jpg') or image.endswith('.JPG')]
    image_list_2 = [ image for image in os.listdir(args.input_dir_2) if image.endswith('.jpg') or image.endswith('.JPG') ]
    output_list = [ image for image in os.listdir(os.path.join(args.output_dir, 'dataset/')) if image.endswith('.jpg') or image.endswith('.JPG') ]
    # print(image_list_2)
    
    # create output path
    os.makedirs(os.path.join(args.output_dir, 'dataset'), exist_ok=True)

    now = datetime.datetime.now()
    new_annotations = dict(
        info=dict(
            description=None,
            url=None,
            version=None,
            year=now.year,
            contributor=None,
            date_created=now.strftime('%Y-%m-%d %H:%M:%S.%f'),
        ),
        licenses=[dict(
            url=None,
            id=0,
            name=None,
        )],
        images=[
            # license, url, file_name, height, width, date_captured, id
        ],
        type='instances',
        annotations=[
            # segmentation, area, iscrowd, image_id, bbox, category_id, id
        ],
        categories=[
            {"supercategory": None, "id": 0, "name": "_background_"}, {"supercategory": None, "id": 1, "name": "clump"}, {"supercategory": None, "id": 2, "name": "stalk"}, {"supercategory": None, "id": 3, "name": "spear"}
        ],
    )

    labels = {1: 'clump', 2: 'stalk' , 3: 'spear'}

    for image_name in tqdm.tqdm(image_list):
        # load image and masks
        if image_name in output_list or image_name == '2625.jpg':
            continue
        image = cv2.imread(os.path.join(args.input_dir, image_name))
        with open(os.path.join(args.input_dir, image_name[:-4]+'.json'), 'r') as jsonfile:
            annotations = json.load(jsonfile)
        annos, width, height = annotations['shapes'], annotations['imageWidth'], annotations['imageHeight']
        masks = annos2masks(annos, width, height)
        
        # random image and masks
        random.shuffle(image_list_2)
        image_2 = cv2.imread(os.path.join(args.input_dir_2, image_list_2[0]))
        print('Current image: ', image_name, '&', image_list_2[0])
        with open(os.path.join(args.input_dir_2, image_list_2[0][:-4]+'.json'), 'r') as jsonfile:
            annotations_2 = json.load(jsonfile)
        annos_2, width_2, height_2 = annotations_2['shapes'], annotations_2['imageWidth'], annotations_2['imageHeight']
        masks_2 = annos2masks(annos_2, width_2, height_2)
        
        # perform copy-paste
        new_image, new_masks = copy_paste(image, masks, image_2, masks_2)
        cv2.imwrite(os.path.join(args.output_dir, 'dataset', image_name), new_image)

        with open(os.path.join(args.output_dir, 'dataset', image_name), 'rb') as img_file:
            imgdata = base64.b64encode(img_file.read()).decode("utf-8")
        content = dict(
            version="4.5.5",
            flags=dict(),
            shapes=[],
            imagePath=image_name,
            imageData=imgdata,
            imageHeight=height,
            imageWidth=width
        )
        for new_anno in masks2annos(new_masks):
            content["shapes"].append(new_anno)
        with open(os.path.join(args.output_dir, image_name[:-4]+'.json'), 'w+') as jsonfile:
            json.dump(content, jsonfile, indent=2)

    

if __name__ == "__main__":
    args = get_args()
    main(args)
