#!/bin/bash
set -e

SRC_NAME="exp021_fold5_Unet++_effb7_advprop_onecycle_mixup_tile336_slide112"

cp "./output/${SRC_NAME}/checkpoint*.pth" "./output/ink-model/"
cp "./output/${SRC_NAME}/last-checkpoint*.pth" "./output/ink-model/"

kaggle datasets version -p ./output/ink-model -m "update: ${SRC_NAME}"
