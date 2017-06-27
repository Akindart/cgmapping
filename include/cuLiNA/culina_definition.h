//
// Created by spades on 01/06/17.
//

#ifndef CGMAPPING_CUBLAS_WRAPPER_DEFINITION_H
#define CGMAPPING_CUBLAS_WRAPPER_DEFINITION_H

#include "cuLiNA/culina_matrix.h"
#define IDX2C(i, j, ld) (((j)*(ld))+(i))

namespace cuLiNA {
    
    typedef culina_matrix<double, 1, 1> culina_scalard;
    typedef culina_matrix<double, 3, 1> culina_vector3d;
    typedef culina_matrix<double, 4, 1> culina_vector4d;
    typedef culina_matrix<double, 6, 1> culina_vector6d;
    typedef culina_matrix<double, 4, 4> culina_matrix4d;
    typedef culina_matrix<double, 3, 3> culina_matrix3d;
    
}

#endif //CGMAPPING_CUBLAS_WRAPPER_DEFINITION_H
