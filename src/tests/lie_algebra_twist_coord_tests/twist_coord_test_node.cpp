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

int main(int argc, char **argv) {

//  ros::init(argc, argv, "eigen_test");
//  ros::NodeHandle nh;
    
    cgmapping::twist6D twist, delta_twist;
    
    cgmapping::Vector6d test_twist;
    
    cuLiNA::culina_matrix3d d_linear_vel, d_angular_vel;
    
    d_linear_vel._allocateMatrixDataMemory();
    d_angular_vel._allocateMatrixDataMemory();
    
    d_linear_vel(0,0) = 0.1;
    d_linear_vel(1,0) = 2;
    d_linear_vel(2,0) = 0;
    
    d_angular_vel(0,0) = 1;
    d_angular_vel(1,0) = 0;
    d_angular_vel(2,0) = 0.2;
    
    
    
    test_twist << 0, 0, 0, 0, 0, 0;
    
    twist._setStateInfo(test_twist);
    
    double t;
    
    
    test_twist << d_linear_vel(0,0), d_linear_vel(1,0), d_linear_vel(2,0),
        d_angular_vel(0,0), d_angular_vel(1,0), d_angular_vel(2,0);
    
    delta_twist._setStateInfo(test_twist);
    cgmapping::Timer tmr;
    twist._update_state(delta_twist, 0.1);
    t = tmr.elapsed();
    d_linear_vel(0,0) = twist._getStateInfo()(0,0);
    d_linear_vel(1,0) = twist._getStateInfo()(1,0);
    d_linear_vel(2,0) = twist._getStateInfo()(2,0);
    
    d_angular_vel(0,0) = twist._getStateInfo()(3,0);
    d_angular_vel(1,0) = twist._getStateInfo()(4,0);
    d_angular_vel(2,0) = twist._getStateInfo()(5,0);
    
    std::cout << "total time " << t * 1000000 << " [us]" << std::endl;
    
    std::cout << twist._getStateInfo().transpose() << std::endl;
    
    return 0;
    
}
