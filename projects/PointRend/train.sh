UDA_VISIBLE_DEVICES=1 python3 train_net.py --config-file ./configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml --num-gpus 1 SOLVER.IMS_PER_BATCH 2 SOLVER>BASE_LR 0.0001
