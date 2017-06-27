//
// Created by spades on 08/06/17.
//

#include <cuda_device_properties/cuda_device_properties.h>

cudaDeviceProp cuda_device_properties::cuda_device_properties::device_prop_;

cudaDeviceProp&cuda_device_properties::cuda_device_properties::_consult_device_properties(){

    return cuda_device_properties::cuda_device_properties::device_prop_;

};

cudaError_t cuda_device_properties::cuda_device_properties::_obtain_from_device_its_properties(){

    int device;
    
    cudaSetDevice(device);
    cudaGetDeviceProperties(&cuda_device_properties::cuda_device_properties::device_prop_, device);

};
