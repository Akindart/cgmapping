//
// Created by spades on 12/04/18.
//

#include <cuLiNA/culina_error_data_types.h>

#include <thrust/device_vector.h>

#include <cuda_device_properties/gpu_cuda_device_properties.h>

#include <cuda_parsing_helper_in_clion/clion_helper.h>

template <typename T, typename Alloc = thrust::device_malloc_allocator<T>>
cuLiNA::cuLiNA_error_t vector_resize(thrust::device_vector<T, Alloc>& vec, size_t new_vec_size, const T value = 0){
    
    vec.resize(new_vec_size, value);
    
    return cuLiNA::CULINA_SUCCESS;
    
};