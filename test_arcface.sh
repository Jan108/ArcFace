#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR='/mnt/data/afarec/code/face_recognition/ArcFace'
DATA_DIR='/mnt/data/afarec/data/PetFace'

for backbone in "r34" "r50"; do
  for loss in "arcface" "cosface"; do
    for cls in "all" "bird" "cat" "dog" "small_animals"; do
      echo "Start testing for ArcFace with ${backbone} and loss ${loss} for class ${cls}"
      PYTHONPATH=$ROOT_DIR:$PYTHONPATH \
        python $ROOT_DIR/evaluation.py \
          --network $backbone \
          --output "${ROOT_DIR}/work_dir/${backbone}_${loss}_${cls}" \
          --weights "${ROOT_DIR}/work_dir/${backbone}_${loss}_${cls}/model.pt" \
          --img_path "${DATA_DIR}/images" \
          --img_verification "${DATA_DIR}/split/${cls}/verification.csv" \
          --img_identification "${DATA_DIR}/split/${cls}/identification_img.csv"
    done
  done
done
