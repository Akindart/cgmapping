//
// Created by spades on 21/02/17.
//

#include "../../include/cgmapping/state6D.h"

bool cgmapping::state6D::_update_state(const cgmapping::state<tf::Transform> &delta_state) {

    //obtaining the transformations from within the state and delta_state objects
    tf::Transform t_state_wrt_world = this->_getStateInfo();
    tf::Transform t_delta_state_wrt_world = delta_state._getStateInfo();

    //getting the rotation matrix from the state and form the delta_state
    tf::Matrix3x3 R_state_wrt_world = t_state_wrt_world.getBasis();
    tf::Matrix3x3 R_delta_state_wrt_world = t_delta_state_wrt_world.getBasis();
    tf::Matrix3x3 R_new_state_wrt_world = R_delta_state_wrt_world;

    //getting the updated state Rnew = Rdelta*Rstate
    R_new_state_wrt_world *= R_state_wrt_world;

    //the origins of both, the origin of the delta_state represents the translation of origin of the current state
    tf::Vector3 o_state_wrt_world = t_state_wrt_world.getOrigin();
    tf::Vector3 o_delta_state_wrt_world = t_delta_state_wrt_world.getOrigin();

    //update of the origin of the state using the translation from the delta_state
    o_state_wrt_world += o_delta_state_wrt_world;

    tf::Quaternion quaternion;

    //getting the quaternion representation from the new axis
    R_new_state_wrt_world.getRotation(quaternion);

    //generation of the new transformation of the state
    tf::Transform newTransfowm(quaternion, o_state_wrt_world);

    //setting the updated state information
    this->_setStateInfo(newTransfowm);

    //if everything occurs ok
    return true;
}
