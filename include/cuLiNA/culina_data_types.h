//
// Created by spades on 14/06/17.
//

#ifndef CGMAPPING_CUBLAS_WRAPPER_DATA_TYPES_H
#define CGMAPPING_CUBLAS_WRAPPER_DATA_TYPES_H

namespace cuLiNA {
    
    typedef enum {
        
        CULINA_SUCCESS, ///<- if everything turns ok
        CULINA_PARAMETERS_MSIMATCH, ///<-if something went wrong with between parameters passed to function
        
    } cuLiNA_error_t;
    
    typedef enum{
        
        CULINA_INVERSE_OFF, ///<- if the matrix is not required to be inverted during operation
        CULINA_INVERSE_ON,  ///<- if the matrix is required to be inverted during operation
        
    }cuLiNA_operation_t;
    
}

#endif //CGMAPPING_CUBLAS_WRAPPER_DATA_TYPES_H
