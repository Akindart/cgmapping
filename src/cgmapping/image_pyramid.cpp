//
// Created by spades on 26/05/17.
//

#include <cgmapping/image_pyramid.h>

cgmapping::image_pyramid::image_pyramid() {

}

cgmapping::image_pyramid::image_pyramid(int row_original_image, int col_original_image, int flags) {
    
    this->d_original_sized_image_.create(row_original_image, col_original_image, flags);
    this->d_halfed_image_.create(row_original_image / 2, col_original_image / 2, flags);
    this->d_quatered_image_.create(row_original_image / 4, col_original_image / 4, flags);
    this->d_octed_image_.create(row_original_image / 8, col_original_image / 8, flags);
    
}

cv::cuda::GpuMat &cgmapping::image_pyramid::_getImageMat(cgmapping::image_size_t image_size) {
    
    if (image_size == cgmapping::ORIGINAL_SIZE)
        return this->d_original_sized_image_;
    else if (image_size == cgmapping::HALF_SIZE)
        return this->d_halfed_image_;
    else if (image_size == cgmapping::QUARTER_SIZE)
        return this->d_quatered_image_;
    else return this->d_octed_image_;
    
}

uchar *cgmapping::image_pyramid::_getImageData(cgmapping::image_size_t image_size) {
    
    if (image_size == cgmapping::ORIGINAL_SIZE)
        return this->d_original_sized_image_.data;
    else if (image_size == cgmapping::HALF_SIZE)
        return this->d_halfed_image_.data;
    else if (image_size == cgmapping::QUARTER_SIZE)
        return this->d_quatered_image_.data;
    else if (image_size == cgmapping::OCT_SIZE)
        return this->d_octed_image_.data;
    else return nullptr;
    
}


int cgmapping::image_pyramid::_generate_pyramid(Mat &src_img, cv::cuda::Stream &strm) {
    
    this->d_original_sized_image_.upload(src_img, strm);
    cv::cuda::pyrDown(this->d_original_sized_image_, this->d_halfed_image_, strm);
    cv::cuda::pyrDown(this->d_halfed_image_, this->d_quatered_image_, strm);
    cv::cuda::pyrDown(this->d_quatered_image_, this->d_octed_image_, strm);
    
    return 0;
    
}