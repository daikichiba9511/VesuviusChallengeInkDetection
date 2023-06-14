#!/bin/bash
set -e

# SRC_NAME="exp028_fold5_Unet++_effb7_advprop_gradualwarm_mixup_tile224_slide74_sharpen_random_resized_crop"
# SRC_NAME="exp042_fold5_Unet++_effb1_advprop_gradualwarm_mixup_tile224_slide74"
# SRC_NAME="exp047_fold5_Unet++_effb4_advprop_gradualwarm_mixup_tile224_slide74"
SRC_NAME="exp060_stackedUnet"

ls "./output/${SRC_NAME}"
echo "Version: ${SRC_NAME}" >./output/ink-model/version.txt
cp -R ./output/${SRC_NAME}/checkpoint_*.pth ./output/ink-model/
cp -R ./output/${SRC_NAME}/last-checkpoint_*.pth ./output/ink-model/

kaggle datasets version -p ./output/ink-model -m "update: ${SRC_NAME}"
