# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import csv
import tqdm
import pickle
import base64
import json
import numpy as np
from PIL import Image
from skimage import measure, morphology
from pycocotools import mask

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo

# constants
WINDOW_NAME = "COCO detections"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
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
        "--json_output",
        help="Directory for json result output."
        )
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
    return parser

def json_output(output, predictions, filename, path):
	'''
	Save predictions as json file
	'''
	out_filename = os.path.join(output, filename) + '.json'
	with open(path, 'rb') as img_file:
		img_data = base64.b64encode(img_file.read()).decode("utf-8")
	labels = {1: 'clump', 2: 'stalk' , 3: 'spear', 4: 'bar'}
	image_height, image_width = predictions['instances'].image_size
	pred_boxes = np.asarray(predictions["instances"].pred_boxes)
	scores = predictions['instances'].scores.cpu().numpy()
	pred_classes = predictions['instances'].pred_classes.cpu().numpy()
	pred_masks = predictions["instances"].pred_masks.cpu().numpy()

	content = {
	"version": "4.5.5",
	"flags": {},
	"shapes": [
	],
	"imagePath": filename,
	"imageData": img_data,
	"imageHeight": image_height,
	"imageWidth": image_width
	}

	for i in range(len(pred_classes)):
		if pred_classes[i] == 1:
			bbox = pred_boxes[i].cpu().numpy().tolist()
			shape ={
			"label": "clump",
			"points": [[bbox[0], bbox[1]], [bbox[2], bbox[3]]],
			"group_id": i,
			"shape_type": "rectangle",
			"flags": {}
			}
			content["shapes"].append(shape)
		else:
			segmentation = pred_masks[i]
			segmentation = measure.find_contours(segmentation.T, 0.5)
			for j in range(len(segmentation)):
				seg = segmentation[j].tolist()
				if len(seg) > 30:
					new_seg = []
					for k in range(0, len(seg), int(len(seg)/30)):
						new_seg.append(seg[i])
					seg = new_seg
				shape = {
				"label": labels[pred_classes[i]],
				"points": seg,
				"group_id": i,
				"shape_type": "polygon",
				"flags": {}
				}
				content["shapes"].append(shape)
	with open(out_filename, 'w+') as jsonfile:
		json.dump(content, jsonfile, indent=2)

def mask2skeleton(pred_mask):
    """
    Convert pred_masek into skeleton.

    Parameters:
        pred_mask(list): 2d list of boolean, Fasle for bg, True for fg.

    Returns:
        skeleton(list): 2d list of 0 and 1, o for bg, 1 for skeleton
    """
    pred_mask = np.asarray(pred_mask) + 0
    skeleton = morphology.skeletonize(pred_mask)
    return skeleton

def csv_out(predictions, filename, path):
    labels = {1: 'clump', 2: 'stalk' , 3: 'spear'}
    image_height, image_width = predictions['instances'].image_size
    pred_boxes = np.asarray(predictions["instances"].pred_boxes)
    scores = predictions['instances'].scores.cpu().numpy()
    pred_classes = predictions['instances'].pred_classes.cpu().numpy()
    pred_masks = predictions["instances"].pred_masks.cpu().numpy()
    table = []
    for i in range(len(pred_classes)):
        if pred_classes[i] == 3:
            boxs = pred_boxes[i].cpu().numpy().tolist()
            skeleton = mask2skeleton(pred_masks[i]).tolist()
            length_skeleton = np.count_nonzero(skeleton)
            length_box = boxs[3] - boxs[1]
            table.append([i, boxs[0], boxs[1], boxs[2], boxs[3], mask.encode(np.asfortranarray(pred_masks[i])), length_skeleton, length_box])
    table = sorted(table, key = lambda x: x[1])
    for i in range(len(table)):
        table[i][0] = i
    table = [['id', 'box(xmin)', 'box(ymin)', 'box(xmax)', 'box(ymax)', 'mask', 'length(skeleton)', 'length(box)']] + table
    with open(os.path.join(path, filename) + '.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(table)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img)
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )

            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                visualized_output.save(out_filename)
                filename = path.split('/')[-1][:-4]
                #csv_out(predictions, filename, args.json_output)
                #print(predictions)
                if args.json_output:
                    json_output(args.json_output, predictions, filename, path)
                #with open('inference_result.txt', 'wb') as resultfile:
                #    for 
                #print(np.asarray(predictions["instances"].pred_boxes[0])[0].cpu().numpy())
                #print(len(predictions["instances"].pred_boxes[0]))
                #print(type(np.asarray(predictions["instances"].pred_boxes[0])[0].cpu().numpy()))
                

                #print(predictions['instances'].pred_masks.cpu().numpy()[1])
                #testmask = predictions['instances'].pred_masks.cpu().numpy()[1]
                #print(type(testmask))
                #testseg = measure.find_contours(testmask, 0.5)
                #print(len(testseg))
                #print(type(testseg))
                #print(type(testseg[0]))
                #print(np.asarray(predictions["instances"].pred_boxes)[1].cpu().numpy().tolist())
                #with open(path, 'rb') as img_file:
                #	print(type(base64.b64encode(img_file.read()).decode("utf-8")))
                #	base64.b64encode(img_file.read()).decode("utf-8")[:100]


            else:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit
    elif args.webcam:
        assert args.input is None, "Cannot have both --input and --webcam!"
        assert args.output is None, "output not yet supported with --webcam!"
        cam = cv2.VideoCapture(0)
        for vis in tqdm.tqdm(demo.run_on_video(cam)):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cam.release()
        cv2.destroyAllWindows()
    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)

        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + ".mkv"
            else:
                output_fname = args.output
            assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*"x264"),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.video_input)
        for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
            if args.output:
                output_file.write(vis_frame)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()
