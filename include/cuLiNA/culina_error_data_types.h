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

#include <iostream>
#include <cublas_v2.h>

namespace cuLiNA {
    
    typedef enum {
        
        CULINA_SUCCESS, ///<- if everything turns ok
        CULINA_PARAMETERS_MISMATCH, ///<-if something went wrong with between parameters passed to function
        
    } cuLiNA_error_t;
    
    
    inline std::string culinaGetErrorString(cuLiNA_error_t& stat){
        
        switch(stat){
            
            case CULINA_SUCCESS: return "CULINA_SUCCESS";
            case CULINA_PARAMETERS_MISMATCH: return "CULINA_PARAMETERS_MISMATCH";
            default: return "UNKOWN ERROR";
            
        }
        
    }
    
    inline void cuLiNACheckErrors(cuLiNA_error_t &stat, const std::string &file, const std::string &function) {
        
        if (stat != CULINA_SUCCESS) {
            
            std::cerr << std::endl << "###########################################################################"
                      << std::endl << std::endl;
            
            std::cerr << "ERROR HAPPENED FROM WITHIN " << function << std::endl;
            std::cerr << "File: \"" << file << "\"." << std::endl;
            std::cerr << "CUDA ERROR: " << culinaGetErrorString(stat) << std::endl;
            
            std::cerr << std::endl << "###########################################################################"
                      << std::endl;
            
        }
        
    }
    
}

#endif //CGMAPPING_CUBLAS_WRAPPER_DATA_TYPES_H
