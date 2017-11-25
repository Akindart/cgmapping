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
    
    typedef enum cuLiNA_error_t{
        
        CULINA_SUCCESS, ///<- if everything turns ok
        CULINA_PARAMETERS_MISMATCH, ///<-if something went wrong with between parameters passed to function
        CULINA_MATRIX_NOT_INSTANTIATED, ///<-if a culina matrix is not instatiated
        CULINA_CUSOLVER_WRAPPER_PROBLEM, ///<- if some part that depends on cusolver fails
        CULINA_INVALID_PARAMETER, ///<- if some parameter needed by a functions has an invalid value
        
    } cuLiNA_error_t;
    
    
    inline std::string culinaGetErrorString(cuLiNA_error_t& stat){
        
        switch(stat){
            
            case CULINA_SUCCESS: return "CULINA_SUCCESS";
            case CULINA_PARAMETERS_MISMATCH: return "CULINA_PARAMETERS_MISMATCH";
            case CULINA_MATRIX_NOT_INSTANTIATED: return "CULINA_MATRIX_NOT_INSTATIATED";
            case CULINA_CUSOLVER_WRAPPER_PROBLEM: return "CULINA_CUSOLVER_WRAPPER_PROBLEM";
            case CULINA_INVALID_PARAMETER: return "CULINA_INVALID_PARAMETER";
            default: return "UNKOWN ERROR";
            
        }
        
    }
    
    inline void cuLiNACheckErrors(cuLiNA_error_t &stat, const std::string &file, const std::string &function, const int line = 0) {
        
        if (stat != CULINA_SUCCESS) {
            
            std::cerr << std::endl << "###########################################################################"
                      << std::endl << std::endl;
            
            std::cerr << "ERROR HAPPENED FROM WITHIN " << function << std::endl;
            std::cerr << "File: \"" << file << "\"." << std::endl;
            std::cerr << "Function: \"" << function << "\"." << std::endl;
            if(line != 0)
                std::cerr << "Line: \"" << line << "\"." << std::endl;
            std::cerr << "CULINA ERROR: " << culinaGetErrorString(stat) << std::endl;
           
            
            
            std::cerr << std::endl << "###########################################################################"
                      << std::endl;
            
        }
        
    }
    
}

#endif //CGMAPPING_CUBLAS_WRAPPER_DATA_TYPES_H
