#!/bin/bash

chmod +x pvo_running_script.sh

#parameters oder is
# 1- dataset/top/folder/address
# 2- dataset/rgb/list/of/images/files/inside/dataset/top/folder
# 3- dataset/depth/list/of/images/files/inside/dataset/top/folder
# 4- dataset/groundtruth/file/inside/dataset/top/folder
# 5- dataset/pose/estimation/file/inside/dataset/top/folder
# 6- accpetance convergence error
# 7- max iterations of aoptimization
# 8- degrees of freedom t-student
# 9- accptance convergance for t student
# 10- min img size
# 11- max img size

################################# INPUT INFORMATION ###############################################################

DATASETS_FOLDER=/home/spades/kinetic_ws/src/cgmapping/0_non_code_stuff/datasets

DATASET_NAME=rgbd_dataset_freiburg1_desk #this one shall be altered for tests with other datasets

DATASET="$DATASETS_FOLDER/$DATASET_NAME/"
RGB_IMG_FILE_LIST=rgb.txt
DEPTH_IMG_FILE_LIST=depth.txt
GROUNDTRUTH_FILE=groundtruth.txt
POSE_ESTIMATION_FILE_PREFIX=pose_estimation

EPS_ERROR=0.000001
MAX_ITERATIONS=100
DEGREES_OF_FREEDOM=5
EPS_ACCPTANCE_T_STUDENT=0.001
MIN_IMG_SIZE=1
MAX_IMG_SIZE=4

INTERATIONS_INFO_FILE=iterations
RESULTANT_ATE_IMG=image_ate


TEST_SUFFIX=_with_prior

################################# OUTPUT INFORMATION ###############################################################

POSE_ESTIMATION_FILE="$POSE_ESTIMATION_FILE_PREFIX$TEST_SUFFIX.txt"
INTERATIONS_INFO_FILE="$INTERATIONS_INFO_FILE$TEST_SUFFIX.txt"


./pvo_running_script.sh $DATASET $RGB_IMG_FILE_LIST $DEPTH_IMG_FILE_LIST $GROUNDTRUTH_FILE $POSE_ESTIMATION_FILE $EPS_ERROR $MAX_ITERATIONS $DEGREES_OF_FREEDOM $EPS_ACCPTANCE_T_STUDENT $MIN_IMG_SIZE $MAX_IMG_SIZE

#sh /home/spades/kinetic/evaluate_rpe.py --delta 0.1 --fixed_delta --verbose --plot image_rpe "$DATASET$GROUNDTRUTH_FILE" POSE_ESTIMATION_FILE

GROUNDTRUTH_ABS_PATH="$DATASET$GROUNDTRUTH_FILE"
POSE_ESTIMATION_ABS_PATH="$DATASET$POSE_ESTIMATION_FILE"
ABSOLUTE_RESULTANT_ATE_IMG_PATH="$DATASET$RESULTANT_ATE_IMG$DATASET_DATASET_NAME$TEST_SUFFIX"

echo $GROUNDTRUTH_ABS_PATH
echo $POSE_ESTIMATION_ABS_PATH
echo $ABSOLUTE_RESULTANT_ATE_IMG_PATH



/home/spades/kinetic_ws/evaluate_ate.py --verbose --plot $ABSOLUTE_RESULTANT_ATE_IMG_PATH $GROUNDTRUTH_ABS_PATH $POSE_ESTIMATION_ABS_PATH

display "$ABSOLUTE_RESULTANT_ATE_IMG_PATH.png"