//
// Created by spades on 21/05/17.
//

#include <image_pyramid_test_cuda.cuh>
#include <cuda.h>
#include <cuda_runtime.h>

int image_processing(){

  Mat h_src, h_src2;

  cv::cuda::Stream strm1, strm2;
  h_src = imread("/home/spades/Pictures/arm_robot_test.jpg", CV_8U);
  h_src2 = imread("/home/spades/Pictures/arm_robot_test.jpg", CV_16U);
//  if(!h_src.data){
//
//    printf("shit happens\n");
//    return -1;
//
//  }

  cuda::GpuMat d_src, d_src2, d_result_strm1, d_result_strm2;

  cuda::GpuMat d_intensity_half, d_intensity_quarter, d_intensity_oct;
  cuda::GpuMat d_depth_half, d_depth_quarter, d_depth_oct;

  Mat h_result, h_result_2;

  d_src.upload(h_src, strm1);
  cuda::pyrDown(d_src, d_intensity_half, strm1);
  d_src2.upload(h_src2, strm2);
  cuda::pyrDown(d_src2, d_depth_half, strm2);
  cuda::pyrDown(d_intensity_half, d_intensity_quarter, strm1);
  cuda::pyrDown(d_intensity_quarter, d_intensity_oct, strm1);
  cuda::pyrDown(d_depth_half, d_depth_quarter, strm2);
  d_intensity_oct.download(h_result, strm1);
  cuda::pyrDown(d_depth_quarter, d_depth_oct, strm2);

  d_depth_oct.download(h_result_2, strm2);

//  std::string window1("Pyramid Demo original");
//  std::string window2("Pyramid Demo enlarged");
//
//      namedWindow(window1, CV_WINDOW_AUTOSIZE);
//      namedWindow(window2, CV_WINDOW_AUTOSIZE);
//
//
//
//
//  imshow(window1, h_src);
//  imshow(window2, h_src2);
//
//      /// Loop
//  while( true ){
//
//      	int c;
//      	c = waitKey(10);
//
//      	if((char) c == 27) break;
//
//   }

  return 0;

}

int device_to_host_test(){

  double h_twist[6] = {1.0, 2.0, 3.0, 4.00, 3.0, 4.0};
  double *d_twist;

  size_t  size;

  size = 6 * sizeof(double);

  std::cout << "hi" << std::endl;

  cudaMalloc(&d_twist, size);
  cudaMemcpy(d_twist, h_twist, size, cudaMemcpyHostToDevice);
  cudaMemcpy(h_twist, d_twist, size, cudaMemcpyDeviceToHost);

  return 0;

}