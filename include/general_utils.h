//
// Created by spades on 29/06/17.
//

#ifndef CGMAPPING_GENERAL_UTILS_H
#define CGMAPPING_GENERAL_UTILS_H

#include <iostream>
#include <cuLiNA/cuBLAS_wrapper/cublas_wrapper.h>
#include <cuLiNA/cuSOLVER_wrapper/cusolver_wrapper.h>

inline void cudaCheckErrors(cudaError_t stat, const std::string& file, const std::string& function){
    
    if(stat != cudaSuccess) {
        
        std::cerr << std::endl << "###########################################################################"
                  << std::endl << std::endl;
        
        std::cerr << "ERROR HAPPENED FROM WITHIN " << function << std::endl;
        std::cerr << "File: \"" << file << "\"." << std::endl;
        std::cerr << "CUDA ERROR: " << cudaGetErrorString(stat) << std::endl;
        
        std::cerr << std::endl << "###########################################################################"
                  << std::endl;
        
    }
    
};

inline void cusolverCheckErrors(cusolverStatus_t stat, const std::string& file, const std::string& function){
    
    if(stat != CUSOLVER_STATUS_SUCCESS) {
        
        
        std::cerr << std::endl << "###########################################################################"
                  << std::endl << std::endl;
        
        std::cerr << "ERROR HAPPENED FROM WITHIN " << function << std::endl;
        std::cerr << "File: \"" << file  << "\"." << std::endl;
        std::cerr << "CUDA ERROR: " << cuSOLVER_wrapper::cusolver_wrapper::_cusolver_wrapper_get_cusolver_error(stat) << std::endl;
        
        std::cerr << std::endl << "###########################################################################"
                  << std::endl;
        
    }
    
};
inline void cublasCheckErrors(cublasStatus_t stat, const std::string& file, const std::string& function){
    
    if(stat != CUBLAS_STATUS_SUCCESS) {
        
        std::cerr << std::endl << "###########################################################################"
                  << std::endl << std::endl;
        
        std::cerr << "ERROR HAPPENED FROM WITHIN " << function << std::endl;
        std::cerr << "File: \"" << file << "\"." << std::endl;
        std::cerr << "CUDA ERROR: " << "de" << std::endl;
        
        std::cerr << std::endl << "###########################################################################"
                  << std::endl;
        
    }
    
};

#endif //CGMAPPING_GENERAL_UTILS_H
