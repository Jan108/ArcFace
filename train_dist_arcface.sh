#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR='/mnt/data/afarec/code/face_recognition/ArcFace'

for backbone in "resnet18" "resnet34"; do
  for loss in "arcface" "cosface"; do
    for cls in "all" "bird" "cat" "dog" "small_animals"; do
      echo "Start training for ArcFace with ${backbone} and loss ${loss} for class ${cls}"
      PYTHONPATH=$ROOT_DIR:$PYTHONPATH \
        CUDA_VISIBLE_DEVICES=0,1 \
        torchrun --nproc_per_node=8 $ROOT_DIR/train_v2.py "${ROOT_DIR}/configs/${backbone}_${loss}_${cls}.py"
    done
  done
done
