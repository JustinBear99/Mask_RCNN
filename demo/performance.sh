CUDA_VISIBLE_DEVICES=1 python3 demo.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml --input ../datasets/coco/annotations/test_637/JPEGImages/*.jpg --output ./performance --json_output ./performance --opts MODEL.WEIGHTS ../output/model_0269999.pth