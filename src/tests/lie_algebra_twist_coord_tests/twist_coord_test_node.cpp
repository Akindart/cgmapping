//
// Created by spades on 20/02/17.
//

#include <ros/ros.h>
#include <iostream>
#include <chrono>
//opencv stuff
#include <Eigen/Dense>
#include "cgmapping/twist6D.h"
#include "cgmapping/timer.h"

using namespace Eigen;

int main(int argc, char **argv){

//  ros::init(argc, argv, "eigen_test");
//  ros::NodeHandle nh;

  cgmapping::twist6D twist, delta_twist;

  cgmapping::Vector6d test_twist;

  test_twist << 0.1,2,0,1,0,0.2;

  delta_twist._setStateInfo(test_twist);

  test_twist << 0,0,0,0,0,0;

  twist._setStateInfo(test_twist);

  double t;
  cgmapping::Timer tmr;

  twist._update_state(delta_twist);

  t = tmr.elapsed();
  std::cout << "total time "  << t*1000000 << " [us]" << std::endl;

  std::cout << twist._getStateInfo().transpose() << std::endl;

  return 0;

}
