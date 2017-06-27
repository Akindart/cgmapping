//
// Created by spades on 21/02/17.
//

#ifndef CGMAPPING_MAP3D_H
#define CGMAPPING_MAP3D_H

#include "map.h"
#include <octomap/OcTree.h>
#include <octomap/Pointcloud.h>

namespace cgmapping {

    class map3D : public map<octomap::OcTree, octomap::Pointcloud> {

    public:

        virtual bool _update_map(const octomap::Pointcloud &delta_map);

    };

}


#endif //CGMAPPING_MAP3D_H
