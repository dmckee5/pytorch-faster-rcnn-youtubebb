#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
TRAIN_DATASET=$2
TEST_DATASET=$3
NET=$4

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:4:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case ${TRAIN_DATASET} in
  pascal_voc)
    TRAIN_IMDB="voc_2007_trainval"
#    TEST_IMDB="voc_2007_test"
    ITERS=70000
    ANCHORS="[8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  pascal_voc_0712)
    TRAIN_IMDB="voc_2007_trainval+voc_2012_trainval"
#    TEST_IMDB="voc_2007_test"
    ITERS=110000
    ANCHORS="[8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  coco)
    TRAIN_IMDB="coco_2014_train+coco_2014_valminusminival"
#    TEST_IMDB="coco_2014_minival"
    ITERS=490000
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  coco23)
    TRAIN_IMDB="coco23_2014_train+coco23_2014_valminusminival"
#    TEST_IMDB="coco23_2014_minival"
    ITERS=490000
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  youtubebb)
    TRAIN_IMDB="youtubebb_2017_train"
#    TEST_IMDB="youtubebb_2017_test"
    ITERS=1250000
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  *)
    echo "No train dataset given"
    exit
    ;;
esac

case ${TEST_DATASET} in
  pascal_voc)
    TEST_IMDB="voc_2007_test"
    ;;
  pascal_voc_0712)
    TEST_IMDB="voc_2007_test"
    ;;
  coco)
    TEST_IMDB="coco_2014_minival"
    ;;
  coco23)
    TEST_IMDB="coco23_2014_minival"
    ;;
  coco23_adapted_instance_2018-01-20-22-02-44)
    TEST_IMDB="coco23_adapted_2014_minival_instance_2018-01-20-22-02-44"
    ;;
  coco23_adapted_instance_2018-01-20-22-02-18)
    TEST_IMDB="coco23_adapted_2014_minival_instance_2018-01-20-22-02-18"
    ;;
  youtubebb)
    TEST_IMDB="youtubebb_2017_test"
    ;;
  *)
    echo "No test dataset given"
    exit
    ;;
esac


LOG="experiments/logs/test_${NET}_${TRAIN_IMDB}_${TEST_IMDB}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

set +x
if [[ ! -z  ${EXTRA_ARGS_SLUG}  ]]; then
  NET_FINAL=output/${NET}/${TRAIN_IMDB}/${EXTRA_ARGS_SLUG}/${NET}_faster_rcnn_iter_${ITERS}.pth
else
  NET_FINAL=output/${NET}/${TRAIN_IMDB}/default/${NET}_faster_rcnn_iter_${ITERS}.pth
fi
set -x

if [[ ! -z  ${EXTRA_ARGS_SLUG}  ]]; then
  CUDA_VISIBLE_DEVICES=${GPU_ID} time python ./tools/test_net.py \
    --imdb ${TEST_IMDB} \
    --model ${NET_FINAL} \
    --cfg experiments/cfgs/${NET}.yml \
    --tag ${EXTRA_ARGS_SLUG} \
    --net ${NET} \
    --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS} \
          ${EXTRA_ARGS}
else
  CUDA_VISIBLE_DEVICES=${GPU_ID} time python ./tools/test_net.py \
    --imdb ${TEST_IMDB} \
    --model ${NET_FINAL} \
    --cfg experiments/cfgs/${NET}.yml \
    --net ${NET} \
    --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS} \
          ${EXTRA_ARGS}
fi


