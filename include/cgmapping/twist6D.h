//
// Created by spades on 19/05/17.
//

#ifndef CGMAPPING_STATE6D_H
#define CGMAPPING_STATE6D_H

#include "cgmapping/state.h"
#include "cgmapping/se3_lie_algebra_utils.cuh"

using namespace Eigen;

typedef Matrix<double, 6, 1> Vector6d;

namespace cgmapping {

    class twist6D : public state<Vector6d> {

    public:

        virtual bool _update_state(const state<Vector6d> &delta_state);

        bool _update_state(const state<Vector6d> &delta_state, double time_elapsed);
    
    };

}

#endif //CGMAPPING_STATE6D_H
