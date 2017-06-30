//
// Created by spades on 01/06/17.
//

#ifndef CGMAPPING_CUBLAS_WRAPPER_DEFINITION_H
#define CGMAPPING_CUBLAS_WRAPPER_DEFINITION_H

#include "cuLiNA/culina_matrix.h"
#include <cuLiNA/culina_error_data_types.h>
#include <cuLiNA/culina_operation_data_types.h>
#define IDX2C(i, j, ld) (((j)*(ld))+(i))

namespace cuLiNA {
    
    ///Commomly used matrices and vectors types
    typedef culina_matrix<double, 1, 1> culina_scalard;
    typedef culina_matrix<double, 3, 1> culina_vector3d;
    typedef culina_matrix<double, 4, 1> culina_vector4d;
    typedef culina_matrix<double, 6, 1> culina_vector6d;
    typedef culina_matrix<double, 4> culina_matrix4d;
    typedef culina_matrix<double, 3> culina_matrix3d;
    
    ///Definitions for the structs containing multiplication parameters
    typedef cuLiNA_operation_parameters_t<float> culiopS_t;
    typedef cuLiNA_operation_parameters_t<double> culiopD_t;
    
    /**
     *
     * The following static declarations contain the standard definitions for the struct that holds
     * the parameters needed to carryout with matrix multiplication. Useful when nothing special is needed.
     *
     */
    static culiopD_t culiopD_default;
    static culiopS_t culiopS_default;
    
}

#endif //CGMAPPING_CUBLAS_WRAPPER_DEFINITION_H
