//
// Created by spades on 08/06/17.
//

#include <cuda_device_properties/cuda_device_properties.h>

cudaDeviceProp cuda_device_properties::device_prop_;
int cuda_device_properties::device_ = -1;

cudaError_t cuda_device_properties::_obtain_from_device_its_properties(){
    
    if(device_ == -1){
        
        device_ = 0;
        cudaSetDevice(device_);
        
    }
    
    return cudaGetDeviceProperties(&device_prop_, device_);
    
};

cudaDeviceProp& cuda_device_properties::_consult_device_properties(){
    
    if(device_ ==  -1){
        
        cuda_device_properties::_obtain_from_device_its_properties();
        
    }
    
    return device_prop_;

};


