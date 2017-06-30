//
// Created by spades on 14/06/17.
//

/*
 *
 * This file contains only data types that are used by other pieces of this code.
 * Note that here no definition of template is done.
 *
 *
 * */

#ifndef CGMAPPING_CUBLAS_WRAPPER_DATA_TYPES_H
#define CGMAPPING_CUBLAS_WRAPPER_DATA_TYPES_H

#include <cublas_v2.h>

namespace cuLiNA {
    
    typedef enum {
        
        CULINA_SUCCESS, ///<- if everything turns ok
        CULINA_PARAMETERS_MSIMATCH, ///<-if something went wrong with between parameters passed to function
        
    } cuLiNA_error_t;
    
}

#endif //CGMAPPING_CUBLAS_WRAPPER_DATA_TYPES_H
