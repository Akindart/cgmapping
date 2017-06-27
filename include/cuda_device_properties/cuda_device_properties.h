//
// Created by spades on 08/06/17.
//

#ifndef CGMAPPING_CUDA_PROPERTIES_H
#define CGMAPPING_CUDA_PROPERTIES_H

#include <cuda_runtime.h>
#include <driver_types.h>

namespace cuda_device_properties {
    
    class cuda_device_properties {
        
        static cudaDeviceProp device_prop_;
     
     public:
        
        static cudaDeviceProp &_consult_device_properties();
        static cudaError_t _obtain_from_device_its_properties();
        
    };
    
}

#endif //CGMAPPING_CUDA_PROPERTIES_H
