//
// Created by spades on 20/02/17.
//

#include <ros/ros.h>
#include <image_pyramid_test_cuda.cuh>
#include <cuBLAS_wrapper/cublas_wrapper.h>
#include <cuBLAS_wrapper/cublas_matrix.h>
#include <cgmapping/image_pyramid.h>
#include <cgmapping/timer.h>

using namespace cv;

void image_copy_test(cgmapping::image_pyramid& test_mat){

  cgmapping::Timer tmr;

  tmr.reset();
  cgmapping::image_pyramid img_pyramid(480, 640, IMREAD_COLOR);
  double t1 = tmr.elapsed();

  cv::Mat mat;

  mat = imread("/home/spades/Pictures/arm_robot_test.jpg", IMREAD_COLOR);

  tmr.reset();
  img_pyramid._generate_pyramid(mat);
  double t2 = tmr.elapsed();

  tmr.reset();
  img_pyramid._generate_pyramid(mat);
  double t3 = tmr.elapsed();

  tmr.reset();
  test_mat = img_pyramid;
  double t4 = tmr.elapsed();

  std::cout << "T1 = " << t1*1000000 << " [us]" << std::endl;
  std::cout << "T2 = " << t2*1000000 << " [us]" << std::endl;
  std::cout << "T3 = " << t3*1000000 << " [us]" << std::endl;
  std::cout << "T4 = " << t4*1000000 << " [us]" << std::endl;

}

int main(int argc, char **argv){

//    ros::init(argc, argv, "cudaPyramidTest");
//    ros::NodeHandle nh;

  cgmapping::image_pyramid img_pyr_2;

  image_copy_test(img_pyr_2);

  cv::Mat mat;

  img_pyr_2._getImageMat(cgmapping::OCT_SIZE).download(mat);

  cv::namedWindow("huehue", IMREAD_COLOR);

  cv::imshow("huehue", mat);

  while( true ){

    int c;
    c = waitKey(10);

    if((char) c == 27) break;

   }


  return 0;
  //return image_processing();
  //return device_to_host_test();

}
