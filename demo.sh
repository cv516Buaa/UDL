export GPUS=1
export GPU_NUM=1
export CHECKPOINT=checkpoint/UDL.pth
export MODEL_CONFIG=configs/udl.yaml

CUDA_VISIBLE_DEVICES=${GPUS} python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} \
    test_net_omnilabel.py \
    --config-file ${MODEL_CONFIG} \
    --weight ${CHECKPOINT} \
    --task_config configs/omnilabel_data.yaml \
    --chunk_size 20 \
    OUTPUT_DIR OUT/${MODEL_NAME} \
    TEST.IMS_PER_BATCH ${GPU_NUM} \
    DATASETS.TEST "('omnilabel_test',)"
