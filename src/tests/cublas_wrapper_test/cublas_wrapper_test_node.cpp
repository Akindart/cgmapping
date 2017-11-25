//
// Created by spades on 01/06/17.
//

#include <iostream>
#include <cuLiNA/culina_definition.h>
#include <cgmapping/timer.h>
#include <cuLiNA/culina_operations.h>
#include <cgmapping/se3_lie_algebra_utils.cuh>


int main(int argc, char **argv) {
    
    
    double t;
    cgmapping::Timer tmr;
    
    cuLiNA::culina_matrix3d d_vector;
    cuLiNA::culina_matrix3d d_vector2;
    cuLiNA::culina_matrix3d d_result;
    cuLiNA::culina_vector3d d_norm_test;
    t = tmr.elapsed();

    //cuda_device_properties_getter::cuda_device_properties_getter cudaDeviceProperties;

    //cuda_device_properties_getter::_obtain_from_device_its_properties();

    std::cout << "Declaration time in us " << t * 1000000 << std::endl;

    double a[3*3] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    //double b[3] = { 6.0, 15.0, 4.0};
    double *c;
    double *b;
    double vectorT[3] = {2.0, 2.0, 2.0};
    size_t size_vec;
    int ld_d_vector;

    //std::cout << size_vec << std::endl;

    ld_d_vector = d_vector._getLeading_dimension();

    size_vec = (uint) d_vector._getRows() * d_vector._getColumns();

    //a = (double *) malloc(size_vec * sizeof(*a));
    b = (double *) malloc(4 * sizeof(*b));
    c = (double *) malloc(sizeof(*c));

//    std::cout << std::endl;
//
//    for (int j = 0; j < d_vector._getRows(); j++) {
//        for (int i = 0; i < d_vector._getColumns(); i++) {
//            a[IDX2C(i, j, ld_d_vector)] = (double) (IDX2C(i, j, ld_d_vector)+1);
//            std::cout << a[IDX2C(j, i, ld_d_vector)] << "\t" << std::endl;
//        }
//
//        std::cout << std::endl;
//
//    }

    std::cout << std::endl;

    for (int j = 0; j < d_vector2._getColumns(); j++) {
        for (int i = 0; i < d_vector2._getRows(); i++) {

            if(i == j)
                b[IDX2C(j, i, ld_d_vector)] = 1.0;
            else b[IDX2C(j, i, ld_d_vector)] = 0.0;

            std::cout << b[IDX2C(j, i, ld_d_vector)] << "\t";
        }

        std::cout << std::endl;

    }

    double result;

    tmr.reset();
    std::cout << "before allocating " << std::endl;
    d_vector._allocateMatrixDataMemory();
    d_vector2._allocateMatrixDataMemory();
    d_result._allocateMatrixDataMemory();
    d_norm_test._allocateMatrixDataMemory();
    //d_result._getData()[0] = 29;
    t = tmr.elapsed();

    d_norm_test._getData().clear();

    //d_result._getData().device_vector;

    for (int j = 0; j <  d_norm_test._getRows(); j++) {
        for (int i = 0; i <  d_norm_test._getColumns(); i++) {
            std::cout << " d_norm_test[" << i << "," << j << "] = " <<  d_norm_test._getData()[IDX2C(j, i, 3)]
                      << "\t";
        }

        std::cout << std::endl;

    }

    std::cout << "allocation time in us " << t * 1000000 << std::endl;
    std::cout << "size of result " << sizeof(*d_result._getRawData()) << std::endl;
    //std::cout << "d_result first element: " << d_result._getData()[0] << std::endl;
    
    cuSOLVER_wrapper::cusolver_wrapper::_start_cusolverDn_handle_wrapper();
    cuBLAS_wrapper::cublas_wrapper::_start_cublas_handle_wrapper();
    //cublasSetPointerMode_v2(cuBLAS_wrapper::cublas_wrapper::_getCublas_handle(), CUBLAS_POINTER_MODE_DEVICE);
    
    //cublas_wrapper::_cublas_set_matrix<double>(a, d_vector);
    d_norm_test._loadData(vectorT);
    d_vector._loadData(a);
    d_vector2._loadData(b);
    
    cuLiNA::culina_Dnorm(&d_norm_test, &result);

    std::cout << "Norm " << result;

    std::cout << std::endl;
    std::cout << std::endl;

    for (int j = 0; j < d_vector._getRows(); j++) {
        for (int i = 0; i < d_vector._getColumns(); i++) {
            a[IDX2C(i, j, ld_d_vector)] = (double) 0;
            std::cout << "d_vector[" << i << "," << j << "] = " << d_vector._getData()[IDX2C(j, i, ld_d_vector)]
                      << "\t";
        }

        std::cout << std::endl;

    }

    std::cout << std::endl;

    for (int j = 0; j < d_vector2._getRows(); j++) {
        for (int i = 0; i < d_vector2._getColumns(); i++) {
            b[IDX2C(i, j, ld_d_vector)] = (double) 0;
            std::cout << "d_vector2[" << i << "," << j << "] = " << d_vector2._getData()[IDX2C(j, i, ld_d_vector)]
                      << std::endl;
        }
    }

    //double alpha = 3;

    int lwork, Lwork;

    //cusolverDnDgeqrf_bufferSize(cuSOLVER_wrapper::cusolver_wrapper::_getCusolverDn_handle(), 4, 4, d_vector._getRawData(), d_vector._getLeading_dimension(),&lwork);

    //std::cout << "lwork: " << lwork << std::endl;

    cuLiNA::culiopD_t culiopD;

    cuLiNA::culina_matrix<double, 24, 1> *workspace = new cuLiNA::culina_matrix<double, 24, 1>();
    workspace->_allocateMatrixDataMemory();

    cudaMalloc ((void**)&culiopD.d_TAU, sizeof(double) * 4);
    cudaMalloc ((void**)&culiopD.dev_info, sizeof(int));

    culiopD.workspace = workspace;
    culiopD.cuLiNA_op_m1 = cuLiNA::CULINA_INVERSE_ON;

    cuLiNA::culina_matrix_Dmultiplication(&d_vector, &d_vector2, NULL, culiopD);



    std::cout << std::endl;
    std::cout << std::endl;

    for (int j = 0; j < d_vector2._getRows(); j++) {
        for (int i = 0; i < d_vector2._getColumns(); i++) {
            std::cout << d_vector2._getData()[IDX2C(j, i, ld_d_vector)] << "\t";
        }

        std::cout << std::endl;

    }

    std::cout << std::endl;

    //cublasStatus_t stat;

    //stat = cuBLAS_wrapper::cublas_wrapper::_cublas_Dmultiplication(d_vector, d_vector2, d_result);

//    if (stat != CUBLAS_STATUS_SUCCESS)
//        std::cout << "shit happens" << std::endl;

    //tmr.reset();

    //d_result._getData().clear();

    //d_result = d_vector2;

    //d_vector2._getData().clear();

    //t=tmr.elapsed();

    // std::cout << "assignment time in us " << t*1000000 << std::endl;

    //cublas_wrapper::_cublas_get_matrix<double>(a, d_vector);
    //d_result._downloadData(c);

#ifdef DEBUG

    for (int j = 0; j < d_result._getRows(); j++) {
        for (int i = 0; i < d_result._getColumns(); i++) {
            printf("%lf \t", c[IDX2C(j, i, ld_d_vector)]);
        }
        printf("\n");
    }

#endif

//    cuLiNA::culina_matrix<double, 4, 2> identity_test;
//
//    identity_test._setIdentity(nullptr);
//
//    printf("\n");
//
//    identity_test._printMatrix();
//
//    cuLiNA::culina_matrix<double, 3, 3> test_m;
//
//    test_m._allocateMatrixDataMemory();
//
//    cusolverDnDgeqrf_bufferSize(cuSOLVER_wrapper::cusolver_wrapper::_getCusolverDn_handle(), 3, 3, test_m._getRawData(), 3, &Lwork );
//
//    std::cout << "QR fact buffersize for 3x3 double matrix: " << Lwork << std::endl;

//    std::cout << "Printing the static identity matrix from definition file" << std::endl;
//
//    for (int j = 0; j < identity4d._getRows(); j++) {
//        for (int i = 0; i < identity4d._getColumns(); i++) {
//            std::cout <<  identity4d._getData()[IDX2C(j, i, identity4d._getLeading_dimension())] << "\t";
//        }
//        printf("\n");
//    }

    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "Now is the lie coordinates test!" << std::endl;
    
    cuLiNA::culina_vector6d d_twist;
    cuLiNA::culina_matrix4d d_homogenic_matrix, d_result_sum;
    cuLiNA::culina_matrix3d d_auxiliar_matrix1, d_auxiliar_matrix2, d_result_sum3;
    cuLiNA::culina_vector3d d_linear_velocity, d_angular_velocity;
    
//    d_homogenic_matrix._allocateMatrixDataMemory();
//    d_auxiliar_matrix1._allocateMatrixDataMemory();
//    d_auxiliar_matrix2._allocateMatrixDataMemory();
//    d_linear_velocity._allocateMatrixDataMemory();
//    d_angular_velocity._allocateMatrixDataMemory();
    
    cuLiNA::cuLiNA_error_t stat = CULINA_SUCCESS;
    
    cuLiNA::cuLiNACheckErrors(stat, __FILE__,__FUNCTION__);
    
    stat = d_auxiliar_matrix1._setIdentity();
    cuLiNA::cuLiNACheckErrors(stat, __FILE__,__FUNCTION__);
    
    stat = d_auxiliar_matrix2._setIdentity();
    cuLiNA::cuLiNACheckErrors(stat, __FILE__,__FUNCTION__);
    
    d_angular_velocity(0,0) = 1;
    d_angular_velocity(1,0) = 0;
    d_angular_velocity(2,0) = 0.2;
    
    d_linear_velocity(0,0) = 0.1;
    d_linear_velocity(1,0) = 2;
    d_linear_velocity(2,0) = 0;
    
    d_twist(3,0) = d_angular_velocity(0,0);
    d_twist(4,0) = d_angular_velocity(1,0);
    d_twist(5,0) = d_angular_velocity(2,0);
    
    d_twist(0,0) = d_linear_velocity(0,0);
    d_twist(1,0) = d_linear_velocity(1,0);
    d_twist(2,0) = d_linear_velocity(2,0);
    
    
    double timeElapsed = 0.1;
    
    cudaStream_t strm;
    cudaStreamCreateWithFlags(&strm, cudaStreamNonBlocking);
    
    //TODO como fazer esta porra para de gerar seg fault apenas por entrar na bendita funcao do mapa exponencial
    tmr.reset();
    cgmapping::cuda::exponential_Dmap_se3(d_linear_velocity,
                                          d_angular_velocity,
                                          d_homogenic_matrix,
                                          d_auxiliar_matrix1,
                                          d_auxiliar_matrix2,
                                          <#initializer#>,
                                          &strm,
                                          nullptr,
                                          nullptr,
                                          0.1);
    t=tmr.elapsed();
    
    std::cout << "Duration: " << t << " [us]" << std::endl;
    
    std::cout << "d_homogenic_matrix = " << std::endl;
    
    d_homogenic_matrix._printMatrix();
    
//    cuLiNA::cuLiNA_error_t culina_stat;
//
//    culina_stat = d_homogenic_matrix._setIdentity();
//
//    cuLiNACheckErrors(culina_stat, __FILE__, __FUNCTION__);
//
//    std::cout << "d_homogenic_matrix after identity op = " << std::endl;
//
//    d_homogenic_matrix._printMatrix();


//    dim3 block_dim, grid_dim;
//
//    cuLiNA::compute_kernel_size_for_matrix_operation(480,640,0,block_dim, grid_dim);
//
//    std::cout << "For a 480x640x1 matrix the kernel config is ";
//
//    std::cout << "block(" << block_dim.x << " ," << block_dim.y << " ," << block_dim.z << ") - ";
//    std::cout << "grid(" << grid_dim.x << " ," << grid_dim.y << " ," << grid_dim.z << ")" << std::endl;
//
//    culiopD.op_m1 = CUBLAS_OP_N;
//    culiopD.op_m2 = CUBLAS_OP_N;
//    culiopD.alpha = 1.0;
//    culiopD.beta = -1.0;
//    culiopD.gamma = 0;
//
//    culina_stat = cuLiNA::culina_matrix_Dsum(&d_homogenic_matrix, &d_homogenic_matrix, &d_result_sum, culiopD);
//
//    cuLiNACheckErrors(culina_stat, __FILE__, __FUNCTION__);
//
//    d_result_sum._printMatrix();
    
    
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "Now comes the logarithmic map test!!!!!!!! HURRAY FOR US!" << std::endl;
    
    tmr.reset();
    cgmapping::cuda::logarithmic_Dmap_se3(d_linear_velocity,
                                          d_angular_velocity,
                                          d_homogenic_matrix,
                                          d_auxiliar_matrix1,
                                          d_auxiliar_matrix2,
                                          &strm,
                                          0.1);
    cudaDeviceSynchronize();
    t=tmr.elapsed();
    
    cudaDeviceSynchronize();
    
    std::cout << "Duration: " << t*1000000 << " [us]" << std::endl;
    
    std::cout << "d_angular_vel = " << std::endl;
    
    d_angular_velocity._printMatrix();
    
    std::cout << "d_linear_vel = " << std::endl;
    
    d_linear_velocity._printMatrix();
    
    return 0;
    
}