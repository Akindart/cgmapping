//
// Created by spades on 21/02/17.
//

#include "cgmapping/map3D.h"

bool cgmapping::map3D::_update_map(const octomap::Pointcloud& delta_map) {

    octomap::OcTree new_map = this->_getMap();

    return true;
}
