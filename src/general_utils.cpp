//
// Created by spades on 29/06/17.
//

#include <general_utils.h>

void cudaCheckErrors(cudaError_t stat, const std::string &file, const std::string &function) {
    
    if (stat != cudaSuccess) {
        
        std::cerr << std::endl << "###########################################################################"
                  << std::endl << std::endl;
        
        std::cerr << "ERROR HAPPENED FROM WITHIN " << function << std::endl;
        std::cerr << "File: \"" << file << "\"." << std::endl;
        std::cerr << "CUDA ERROR: " << cudaGetErrorString(stat) << std::endl;
        
        std::cerr << std::endl << "###########################################################################"
                  << std::endl;
        
    }
    
};