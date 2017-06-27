//
// Created by spades on 21/02/17.
//

#ifndef CGMAPPING_STATE6D_H
#define CGMAPPING_STATE6D_H

#include <tf/tf.h>
#include "state.h"

namespace cgmapping {

    class state6D : public state<tf::Transform> {

    public:

        virtual bool _update_state(const state<tf::Transform> &delta_state);

    };

}

#endif //CGMAPPING_STATE6D_H
