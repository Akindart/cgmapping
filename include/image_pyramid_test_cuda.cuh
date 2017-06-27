//
// Created by spades on 21/05/17.
//

#ifndef CGMAPPING_IMAGE_PYRAMID_TEST_CUDA_H
#define CGMAPPING_IMAGE_PYRAMID_TEST_CUDA_H

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

int image_processing();
int device_to_host_test();
int cuBLAS_test();

#endif //CGMAPPING_IMAGE_PYRAMID_TEST_CUDA_H
