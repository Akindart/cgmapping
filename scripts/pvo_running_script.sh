#!/bin/bash

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

echo $1
echo $2
echo $3
echo $4
echo $5
echo $6
echo $7
echo $8
echo $9
echo ${10}
echo ${11}

#rosrun cgmapping basic_matrix_tests --gtest_filter=visual_odometry_check.visual_odometry_test_with_line_arguments "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$8" "$9" "${10}" "${11}"

rosrun cgmapping basic_matrix_tests --gtest_filter=visual_odometry_check.visual_odometry_test_with_line_arguments $*