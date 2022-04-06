DATA=$1
CUDA_VISIBLE_DEVICES=0 python tools/train.py --root $DATA --trainer FixMatch \
	--source-domains cartoon art_painting photo --target-domains sketch \
	--dataset-config-file configs/datasets/da/pacs.yaml --config-file configs/trainers/da/didanet/pacs_dida_net.yaml \
	--output-dir output/didanet_pacs/sketch \
	MODEL.BACKBONE.NAME resnet18_dida TRAIN.CHECKPOINT_FREQ 10
