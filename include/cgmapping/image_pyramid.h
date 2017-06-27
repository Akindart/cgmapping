//
// Created by spades on 26/05/17.
//

#ifndef CGMAPPING_IMAGE_PYRAMID_H
#define CGMAPPING_IMAGE_PYRAMID_H

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/cudawarping.hpp>

using namespace cv;

namespace cgmapping {

  typedef enum {

    ORIGINAL_SIZE,
    HALF_SIZE,
    QUARTER_SIZE,
    OCT_SIZE

  } image_size_t;

  class image_pyramid {

    cv::cuda::GpuMat d_original_sized_image_;
    cv::cuda::GpuMat d_halfed_image_;
    cv::cuda::GpuMat d_quatered_image_;
    cv::cuda::GpuMat d_octed_image_;


   public:

    image_pyramid();
    image_pyramid(int row_original_image, int col_original_image, int flags);

    cv::cuda::GpuMat& _getImageMat(image_size_t image_size);
    uchar* _getImageData(image_size_t image_size);

    int _generate_pyramid(Mat &src_img);
    int _generate_pyramid(Mat &src_img, cv::cuda::Stream &strm);

  };

}

#endif //CGMAPPING_IMAGE_PYRAMID_H
