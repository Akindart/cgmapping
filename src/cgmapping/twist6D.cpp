//
// Created by spades 19/05/17
//

#include <cgmapping/timer.h>
#include "cgmapping/twist6D.h"

bool cgmapping::twist6D::_update_state(const state<Vector6d> &delta_state){

  //cgmapping::Timer tmr, tmr2;

  Vector6d delta_twist, new_twist, cur_twist;
  Matrix4d homogenic_transformation;

  cur_twist = this->_getStateInfo();
  delta_twist = delta_state._getStateInfo();

  //tmr.reset();

  //homogenic_transformation = exponential_map_se3(cur_twist);

  //double t = tmr.elapsed();

  //std::cout << "time elapsed exponential: " << t*1000000 << " [us]" << std::endl;

  //tmr.reset();

  homogenic_transformation *= exponential_map_se3(delta_twist);

  //t = tmr.elapsed();

  //std::cout << "time elapsed exponential delta: " << t*1000000 << " [us]" << std::endl;

  //tmr.reset();

  new_twist = logarithmic_map_se3(homogenic_transformation);

  //t = tmr.elapsed();

  //std::cout << "time elapsed logarithmic: " << t*1000000  << " [us]" << std::endl;

  //tmr.reset();

  this->_setStateInfo(new_twist);

  //t = tmr.elapsed();

  //std::cout << "set time: " << t*1000000  << " [us]" << std::endl;

  //t = tmr2.elapsed();
  //std::cout << "total inside function: " << t*1000000  << " [us]" << std::endl;

  return true;

}

bool cgmapping::twist6D::_update_state(const state<Vector6d> &delta_state, double time_elapsed){

  Vector6d delta_twist, new_twist, cur_twist;
  Matrix4d homogenic_transformation;

  cur_twist = this->_getStateInfo();
  delta_twist = delta_state._getStateInfo();

  homogenic_transformation = exponential_map_se3(cur_twist, time_elapsed)*exponential_map_se3(delta_twist, time_elapsed);

  new_twist = logarithmic_map_se3(homogenic_transformation, time_elapsed);

  this->_setStateInfo(new_twist);

  return true;

}