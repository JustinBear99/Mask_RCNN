CUDA_VISIBLE_DEVICES=0,1 python3 train_net.py --config-file configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml --num-gpus 2 SOLVER.IMS_PER_BATCH 16 SOLVER.BASE_LR 0.0001
