#!/bin/bash
set -e

SRC_NAME="exp028_fold5_Unet++_effb7_advprop_gradualwarm_mixup_tile224_slide74_sharpen_random_resized_crop"
ls "./output/${SRC_NAME}"
cp -R ./output/${SRC_NAME}/checkpoint_*.pth ./output/ink-model/
cp -R ./output/${SRC_NAME}/last-checkpoint_*.pth ./output/ink-model/

kaggle datasets version -p ./output/ink-model -m "update: ${SRC_NAME}"
