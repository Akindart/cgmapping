//
// Created by spades on 01/06/17.
//

#include <iostream>
#include <cuLiNA/culina_definition.h>
#include "cuLiNA/cuBLAS_wrapper/cublas_wrapper.h"
#include <cgmapping/timer.h>
#include <cuda_device_properties/cuda_device_properties.h>
#include <cuLiNA/culina_operations.h>
#include <cusolverDn.h>
#include <cuLiNA/cuSOLVER_wrapper/cusolver_wrapper.h>


int main(int argc, char **argv) {
    
    double t;
    cgmapping::Timer tmr;
    
    cuLiNA::culina_vector4d d_vector;
    cuLiNA::culina_vector4d d_vector2;
    cuLiNA::culina_scalard d_result;
    t = tmr.elapsed();
    
    cuda_device_properties::cuda_device_properties cudaDeviceProperties;
    
    cudaDeviceProperties._obtain_from_device_its_properties();
    
    std::cout << "Declaration time in us " << t * 1000000 << std::endl;
    
    double *a, *b, *c;
    size_t size_vec;
    int ld_d_vector;
    
    //std::cout << size_vec << std::endl;
    
    ld_d_vector = d_vector._getLeading_dimension();
    
    size_vec = (uint) d_vector._getRows() * d_vector._getColumns();
    
    a = (double *) malloc(size_vec * sizeof(*a));
    b = (double *) malloc(4 * sizeof(*b));
    c = (double *) malloc(sizeof(*c));
    
    for (int j = 0; j < d_vector._getColumns(); j++) {
        for (int i = 0; i < d_vector._getRows(); i++) {
            a[IDX2C(i, j, ld_d_vector)] = (double) (IDX2C(i, j, ld_d_vector));
            printf("%lf \n", a[IDX2C(i, j, ld_d_vector)]);
        }
    }
    
    for (int j = 0; j < d_vector2._getColumns(); j++) {
        for (int i = 0; i < d_vector2._getRows(); i++) {
            b[IDX2C(i, j, ld_d_vector)] = (double) (IDX2C(i, j, ld_d_vector));
            printf("%lf \n", b[IDX2C(i, j, ld_d_vector)]);
        }
    }
    
    tmr.reset();
    d_vector._allocateMatrixDataMemory();
    d_vector2._allocateMatrixDataMemory();
    d_result._allocateMatrixDataMemory();
    //d_result._getData()[0] = 29;
    t = tmr.elapsed();
    
    //d_result._getData().clear();
    
    //d_result._getData().device_vector;
    
    std::cout << "allocation time in us " << t * 1000000 << std::endl;
    std::cout << "size of result " << sizeof(*d_result._getRawData()) << std::endl;
    //std::cout << "d_result first element: " << d_result._getData()[0] << std::endl;
    
    cuSOLVER_wrapper::cusolver_wrapper::_start_cusolverDn_handle_wrapper();
    cuBLAS_wrapper::cublas_wrapper::_start_cublas_handle_wrapper();
    cublasSetPointerMode_v2(cuBLAS_wrapper::cublas_wrapper::_getCublas_handle(), CUBLAS_POINTER_MODE_DEVICE);
    
    //cublas_wrapper::_cublas_set_matrix<double>(a, d_vector);
    d_vector._loadData(a);
    d_vector2._loadData(b);
    
    for (int j = 0; j < d_vector._getColumns(); j++) {
        for (int i = 0; i < d_vector._getRows(); i++) {
            a[IDX2C(i, j, ld_d_vector)] = (double) 0;
            std::cout << "d_vector[" << i << "," << j << "] = " << d_vector._getData()[IDX2C(i, j, ld_d_vector)]
                      << std::endl;
        }
    }
    
    for (int j = 0; j < d_vector2._getColumns(); j++) {
        for (int i = 0; i < d_vector2._getRows(); i++) {
            b[IDX2C(i, j, ld_d_vector)] = (double) 0;
            std::cout << "d_vector2[" << i << "," << j << "] = " << d_vector2._getData()[IDX2C(i, j, ld_d_vector)]
                      << std::endl;
        }
    }
    
    //double alpha = 3;
    
    
    cublasStatus_t stat;
    
    stat = cuBLAS_wrapper::cublas_wrapper::_cublas_Dmultiplication(d_vector, d_vector2, d_result);
    
    if (stat != CUBLAS_STATUS_SUCCESS)
        std::cout << "shit happens" << std::endl;
    
    //tmr.reset();
    
    //d_result._getData().clear();
    
    //d_result = d_vector2;
    
    //d_vector2._getData().clear();
    
    //t=tmr.elapsed();
    
    // std::cout << "assignment time in us " << t*1000000 << std::endl;
    
    //cublas_wrapper::_cublas_get_matrix<double>(a, d_vector);
    d_result._downloadData(c);

#ifdef DEBUG
    
    for (int j = 0; j < d_result._getColumns(); j++) {
        for (int i = 0; i < d_result._getRows(); i++) {
            printf("%lf \t", c[IDX2C(i, j, ld_d_vector)]);
        }
        printf("\n");
    }

#endif
    
    
    
    cuLiNA::culina_matrix<double, 4, 2> identity_test;
    
    identity_test._setIdentity(nullptr);
    
    printf("\n");

    for (int j = 0; j < identity_test._getColumns(); j++) {
        for (int i = 0; i < identity_test._getRows(); i++) {
            std::cout <<  identity_test._getData()[IDX2C(i, j, identity_test._getLeading_dimension())] << "\t";
        }
        printf("\n");
    }
    
    cuLiNA::culina_matrix<double, 3, 3> test_m;
    
    test_m._allocateMatrixDataMemory();
    
    int Lwork;
    cusolverDnDgeqrf_bufferSize(cuSOLVER_wrapper::cusolver_wrapper::_getCusolverDn_handle(), 3, 3, test_m._getRawData(), 3, &Lwork );
    
    std::cout << "QR fact buffersize for 4x4 double matrix: " << Lwork << std::endl;
    
    return 0;
    
}