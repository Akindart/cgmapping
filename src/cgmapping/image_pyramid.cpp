//
// Created by spades on 26/05/17.
//

#include <cgmapping/image_pyramid.h>

cgmapping::image_pyramid::image_pyramid(int row_original_image, int col_original_image, int flags) {
    
    cv::cuda::createContinuous(row_original_image, col_original_image, flags, d_original_sized_image_);
    cv::cuda::createContinuous(row_original_image / 2, col_original_image / 2, flags, d_halfed_image_);
    cv::cuda::createContinuous(row_original_image / 4, col_original_image / 4, flags, d_quatered_image_);
    cv::cuda::createContinuous(row_original_image / 8, col_original_image / 8, flags, d_octed_image_);
    cv::cuda::createContinuous(row_original_image / 16, col_original_image / 16, flags, d_hexadec_image_);
    
    cv::Size tmp_size(5, 5);
    
    this->gaussian_filter = cv::cuda::createGaussianFilter(flags, flags, tmp_size, 0.1);
    
}

cv::cuda::GpuMat &cgmapping::image_pyramid::_getImageMat(cgmapping::image_size_t image_size) {
    
    if (image_size == cgmapping::ORIGINAL_SIZE)
        return this->d_original_sized_image_;
    else if (image_size == cgmapping::HALF_SIZE)
        return this->d_halfed_image_;
    else if (image_size == cgmapping::QUARTER_SIZE)
        return this->d_quatered_image_;
    else if (image_size == cgmapping::OCT_SIZE)
        return this->d_octed_image_;
    else if (image_size == cgmapping::HEX_SIZE)
        return this->d_hexadec_image_;
}

int cgmapping::image_pyramid::_generate_pyramid(const Mat &src_img, cv::cuda::Stream &strm) {
    
    this->d_original_sized_image_.upload(src_img, strm);
    this->gaussian_filter->apply(this->d_original_sized_image_, this->d_original_sized_image_, strm);
    cv::cuda::pyrDown(this->d_original_sized_image_, this->d_halfed_image_, strm);
    this->gaussian_filter->apply(this->d_halfed_image_, this->d_halfed_image_, strm);
    cv::cuda::pyrDown(this->d_halfed_image_, this->d_quatered_image_, strm);
    this->gaussian_filter->apply(this->d_quatered_image_, this->d_quatered_image_, strm);
    cv::cuda::pyrDown(this->d_quatered_image_, this->d_octed_image_, strm);
    this->gaussian_filter->apply(this->d_octed_image_, this->d_octed_image_, strm);
    cv::cuda::pyrDown(this->d_octed_image_, this->d_hexadec_image_, strm);
    this->gaussian_filter->apply(this->d_hexadec_image_, this->d_hexadec_image_, strm);
    
    return 0;
    
}
cgmapping::image_pyramid::~image_pyramid() {
    
    d_original_sized_image_.release();
    d_halfed_image_.release();
    d_quatered_image_.release();
    d_octed_image_.release();
    d_hexadec_image_.release();
    
}
