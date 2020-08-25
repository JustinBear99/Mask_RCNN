CUDA_VISIBLE_DEVICES=0 python3 train_net.py --config-file configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS output/model_0244999.pth
