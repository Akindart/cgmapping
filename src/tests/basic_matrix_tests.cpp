//
// Created by spades on 18/12/17.
//

#define CUDA_API_PER_THREAD_DEFAULT_STREAM

#include <gtest/gtest.h>

#include <cmath>

#include <cuLiNA/culina_matrix.h>
#include <cuLiNA/culina_operations.h>
#include <cuLiNA/culina_definition.h>

#include <cgmapping/se3_lie_algebra_utils.cuh>
#include <cgmapping/timer.h>
#include <cgmapping/cgmapping_utils.h>
#include <cgmapping/image_pyramid.h>
#include <cgmapping/img_motion_estimator.h>
#include <cgmapping/visual_odometer.h>

#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <cv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaarithm.hpp>

#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>

#include <fstream>
#include <iostream>

#include <boost/regex.hpp>
#include <boost/algorithm/string/regex.hpp>

#define IDX2X_CM(i,j,ld) ld*j + i

int global_argc;
char **global_argv;

TEST(sum_check, trans_test14x3){

    cuLiNA::culina_matrix<double, 3, 14> A;
    cuLiNA::culina_matrix<double, 14, 3> B;
    cuLiNA::culina_matrix<double, 14, 3> C;
    cuLiNA::culina_matrix<double, 14, 3> C_from_matlab;

    cuLiNA::culiopD_t culiopD;

    culiopD.alpha = 1.0;
    culiopD.beta = 0;
    culiopD.gamma = 0;
    culiopD.op_m1 = CUBLAS_OP_T;

    cudaStream_t strm;
    cudaStreamCreate(&strm);

    culiopD.strm = &strm;

    std::string A_str("/home/spades/kinetic_ws/src/cgmapping/0_non_code_stuff/datasets/matrices_to_test/sum_test/A_3x14.matrix");
    cuLiNA::culina_load_matrix_file<double>(A, A_str);
    std::string B_str("/home/spades/kinetic_ws/src/cgmapping/0_non_code_stuff/datasets/matrices_to_test/sum_test/B_14x3.matrix");
    cuLiNA::culina_load_matrix_file<double>(B, B_str);
    std::string C_str("/home/spades/kinetic_ws/src/cgmapping/0_non_code_stuff/datasets/matrices_to_test/sum_test/C_14x3.matrix");
    cuLiNA::culina_load_matrix_file<double>(C_from_matlab, C_str);

//    A._printMatrix();
//
//    std::cout << std::endl << std::endl;

//    B._printMatrix();
//
//    std::cout << std::endl << std::endl;

    cuLiNA::cuLiNA_error_t stat = cuLiNA::culina_matrix_Dsum(&A, &B, &C, culiopD);

    cudaDeviceSynchronize();

//    C._printMatrix();
//
//    std::cout << std::endl << std::endl;
//
//    C_from_matlab._printMatrix();
//
//    std::cout << std::endl << std::endl;
    
    cuLiNA::cuLiNACheckErrors(stat, __FILE__, __FUNCTION__, __LINE__);

    EXPECT_TRUE(C == C_from_matlab);

}

TEST(sum_check, acc_test_13x7){

    cuLiNA::culina_matrix<double, 13, 7> A;
    cuLiNA::culina_matrix<double, 13, 7> B;
    cuLiNA::culina_matrix<double, 13, 7> C;
    cuLiNA::culina_matrix<double, 13, 7> C_from_matlab;

    cuLiNA::culiopD_t culiopD;

    culiopD.alpha = 1;
    culiopD.beta = 1;


    cudaStream_t strm;
    cudaStreamCreate(&strm);

    culiopD.strm = &strm;

    std::string A_str("/home/spades/kinetic_ws/src/cgmapping/0_non_code_stuff/datasets/matrices_to_test/sum_test/A_13x7.matrix");
    cuLiNA::culina_load_matrix_file<double>(A, A_str);
    std::string B_str("/home/spades/kinetic_ws/src/cgmapping/0_non_code_stuff/datasets/matrices_to_test/sum_test/B_13x7.matrix");
    cuLiNA::culina_load_matrix_file<double>(B, B_str);
    std::string C_str("/home/spades/kinetic_ws/src/cgmapping/0_non_code_stuff/datasets/matrices_to_test/sum_test/C_13x7.matrix");
    cuLiNA::culina_load_matrix_file<double>(C_from_matlab, C_str);

    cuLiNA::cuLiNA_error_t stat;

    stat = cuLiNA::culina_matrix_Dsum(&A, &B, &B, culiopD);

    cudaDeviceSynchronize();

    EXPECT_TRUE(B == C_from_matlab);

}

TEST(sum_check, sub_test12x3){

    cuLiNA::culina_matrix<double, 12, 3> A;
    cuLiNA::culina_matrix<double, 12, 3> B;
    cuLiNA::culina_matrix<double, 12, 3> C;
    cuLiNA::culina_matrix<double, 12, 3> C_from_matlab;

    cuLiNA::culiopD_t culiopD;

    culiopD.alpha = 1;
    culiopD.beta = -1;

    cudaStream_t strm;
    cudaStreamCreate(&strm);

    culiopD.strm = &strm;

    std::string A_str("/home/spades/kinetic_ws/src/cgmapping/0_non_code_stuff/datasets/matrices_to_test/sum_test/A_12x3.matrix");
    cuLiNA::culina_load_matrix_file<double>(A, A_str);
    std::string B_str("/home/spades/kinetic_ws/src/cgmapping/0_non_code_stuff/datasets/matrices_to_test/sum_test/B_12x3.matrix");
    cuLiNA::culina_load_matrix_file<double>(B, B_str);
    std::string C_str("/home/spades/kinetic_ws/src/cgmapping/0_non_code_stuff/datasets/matrices_to_test/sum_test/C_12x3.matrix");
    cuLiNA::culina_load_matrix_file<double>(C_from_matlab, C_str);

    cuLiNA::cuLiNA_error_t stat;

    stat = cuLiNA::culina_matrix_Dsum(&A, &B, &C, culiopD);

    cudaDeviceSynchronize();

    EXPECT_TRUE(C == C_from_matlab);

}

TEST(sum_check, sum_test6x6){

    cuLiNA::culina_matrix<double, 6, 6> A;
    cuLiNA::culina_matrix<double, 6, 6> B;
    cuLiNA::culina_matrix<double, 6, 6> C;
    cuLiNA::culina_matrix<double, 6, 6> C_from_matlab;

    cuLiNA::culiopD_t culiopD;

    culiopD.alpha = 1;
    culiopD.beta = 1;

    cudaStream_t strm;
    cudaStreamCreate(&strm);

    culiopD.strm = &strm;

    std::string A_str("/home/spades/kinetic_ws/src/cgmapping/0_non_code_stuff/datasets/matrices_to_test/sum_test/A_6x6.matrix");
    cuLiNA::culina_load_matrix_file<double>(A, A_str);
    std::string B_str("/home/spades/kinetic_ws/src/cgmapping/0_non_code_stuff/datasets/matrices_to_test/sum_test/B_6x6.matrix");
    cuLiNA::culina_load_matrix_file<double>(B, B_str);
    std::string C_str("/home/spades/kinetic_ws/src/cgmapping/0_non_code_stuff/datasets/matrices_to_test/sum_test/C_6x6.matrix");
    cuLiNA::culina_load_matrix_file<double>(C_from_matlab, C_str);
    
//    A._printMatrix();
//
//    std::cout << std::endl;
//
//    B._printMatrix();
//
//    std::cout << std::endl;

    cuLiNA::cuLiNA_error_t stat = cuLiNA::culina_matrix_Dsum(&A, &B, &C, culiopD);

    cudaDeviceSynchronize();
    
//    C._printMatrix();
//
//    std::cout << std::endl;
//
//    C_from_matlab._printMatrix();
//
//    std::cout << std::endl;
//
    cuLiNA::cuLiNACheckErrors(stat, __FILE__, __FUNCTION__, __LINE__);

    EXPECT_TRUE(C == C_from_matlab);

}

TEST(skew_check, skew_test){

    cuLiNA::culina_vector3d vec_test, vec_answer;

    vec_test(0,0) = 1;
    vec_test(1,0) = 2;
    vec_test(2,0) = 3;

    cuLiNA::culina_matrix3d mat_test;
    //cuLiNA::culina_matrix4d mat_test;
    cuLiNA::culina_matrix3d mat_answer;
    
    
    for (int i = 0; i < mat_test._getRows(); ++i) {
        for (int j = 0; j < mat_test._getColumns(); ++j) {
            
            mat_answer(i,j) = 0;
            mat_test(i,j) = 0;
            
        }
    }
    
    mat_answer(0,1) = -vec_test(2,0);
    mat_answer(0,2) = vec_test(1,0);
    mat_answer(1,2) = -vec_test(0,0);

    mat_answer(1,0) = vec_test(2,0);
    mat_answer(2,0) = -vec_test(1,0);
    mat_answer(2,1) = vec_test(0,0);

    cuLiNA::culiopD_t culiopD;

    culiopD.alpha = 1;
    culiopD.beta = 0;

    cudaStream_t strm;
    cudaStreamCreate(&strm);

    culiopD.strm = &strm;
    
    auto stat = cuLiNA::culina_Dskew_matrix3x3_operator(&vec_test, &mat_test, culiopD);
    cuLiNACheckErrors(stat, __FILE__, __FUNCTION__, __LINE__);
    
    stat = cuLiNA::culina_Dvector_from_skew_matrix3x3_operator(&mat_test, &vec_answer, culiopD);
    cuLiNACheckErrors(stat, __FILE__, __FUNCTION__, __LINE__);
    
    cudaDeviceSynchronize();
    
    vec_test._printMatrix();
    vec_answer._printMatrix();
    
    EXPECT_TRUE( mat_answer == mat_test && vec_test == vec_answer);

}

TEST(diagonal_check, diagonal_test){

    cuLiNA::culina_vector3d vec_test;
    cuLiNA::culina_vector3d vec_answer;
    cuLiNA::culina_matrix3d mat(cuLiNA::IDENTITY);

    vec_answer(0,0) = 1;
    vec_answer(1,0) = 1;
    vec_answer(2,0) = 1;

    vec_test(0,0) = 0;
    vec_test(1,0) = 0;
    vec_test(2,0) = 0;

    cuLiNA::culiopD_t culiopD;

    culiopD.alpha = 1;
    culiopD.beta = 1;

    cudaStream_t strm;
    cudaStreamCreate(&strm);

    culiopD.strm = &strm;

    cuLiNA::cuLiNA_error_t status;

    status = cuLiNA::culina_Ddiagonal_to_vector_operation(&mat, &vec_test, culiopD);

    cuLiNA::cuLiNACheckErrors(status, __FILE__, __FUNCTION__, __LINE__);

    EXPECT_TRUE(vec_answer == vec_test);


}

TEST(trace_check, trace_test){

    double val_test, val_answer;

    //cuLiNA::culina_matrix<double, 4, 5> mat(cuLiNA::IDENTITY);
    cuLiNA::culina_matrix4d mat(cuLiNA::IDENTITY);
    cuLiNA::culina_vector4d vec;

    val_test = 0;
    val_answer = 4.0;

    cuLiNA::culiopD_t culiopD;

    culiopD.alpha = 1;
    culiopD.beta = 1;

    cudaStream_t strm;
    cudaStreamCreate(&strm);

    culiopD.strm = &strm;

    cuLiNA::cuLiNA_error_t status;

    status = cuLiNA::culina_Dtrace_operation(&mat, &vec, val_test, culiopD);

    cuLiNA::cuLiNACheckErrors(status, __FILE__, __FUNCTION__, __LINE__);

    EXPECT_TRUE(val_test == val_answer);

}

TEST(norm_check, norm_test){

    cuLiNA::culina_vector3d vec_test;

    vec_test(0,0) = 2;
    vec_test(1,0) = 2;
    vec_test(2,0) = 2;

    double val_test = 0;
    double val_answer  = sqrt(12.0);

    cuLiNA::culiopD_t culiopD;

    culiopD.alpha = 1;
    culiopD.beta = 1;

    cudaStream_t strm;
    cudaStreamCreate(&strm);

    culiopD.strm = &strm;

    cuLiNA::cuLiNA_error_t stat;

    stat = cuLiNA::culina_Dnorm(&vec_test, &val_test, culiopD);

    cuLiNA::cuLiNACheckErrors(stat, __FILE__, __FUNCTION__, __LINE__);

    EXPECT_TRUE(val_test == val_answer);

}

TEST(block_check, block5x1in7x2start0x1_test){

    cuLiNA::culina_matrix<double, 5, 1> A;
    cuLiNA::culina_matrix<double, 7, 2> T;
    cuLiNA::culina_matrix<double, 7, 2> T_answer;
    
    for (int i = 0; i < T._getRows(); ++i) {
        for (int j = 0; j < T._getColumns(); ++j) {
            
            T(i,j) = 0;
            
        }
    }
    
    cuLiNA::culiopD_t culiopD;

    culiopD.alpha = 1;

    cudaStream_t strm;
    cudaStreamCreate(&strm);

    culiopD.strm = &strm;

    std::string A_str("/home/spades/kinetic_ws/src/cgmapping/0_non_code_stuff/datasets/matrices_to_test/block_test/A_5x1.matrix");
    cuLiNA::culina_load_matrix_file<double>(A, A_str);
    std::string T_str("/home/spades/kinetic_ws/src/cgmapping/0_non_code_stuff/datasets/matrices_to_test/block_test/T_7x2.matrix");
    cuLiNA::culina_load_matrix_file<double>(T_answer, T_str);

    
    
    cuLiNA::cuLiNA_error_t stat;

    stat = cuLiNA::culina_Dblock_assignment_operation(&A, &T ,0,0,0,1,A._getRows(), A._getColumns(), culiopD);
    
    cudaDeviceSynchronize();
    
    A._printMatrix();
    
    std::cout << std::endl;
    
    T_answer._printMatrix();
    
    std::cout << std::endl;
    
    T._printMatrix();
    
    cuLiNA::cuLiNACheckErrors(stat, __FILE__, __FUNCTION__, __LINE__);

    EXPECT_TRUE( T == T_answer);

}

TEST(block_check, block2x2in5x5start0x0_test){

    cuLiNA::culina_matrix<double, 2, 2> A;
    cuLiNA::culina_matrix<double, 5, 5> T;
    cuLiNA::culina_matrix<double, 5, 5> T_answer;

    cuLiNA::culiopD_t culiopD;

    culiopD.alpha = 1;

    cudaStream_t strm;
    cudaStreamCreate(&strm);

    culiopD.strm = &strm;

    for (int i = 0; i < T._getRows(); ++i) {
        for (int j = 0; j < T._getColumns(); ++j) {

            T(i,j) = 0;

        }
    }

    std::string A_str("/home/spades/kinetic_ws/src/cgmapping/0_non_code_stuff/datasets/matrices_to_test/block_test/A_2x2.matrix");
    cuLiNA::culina_load_matrix_file<double>(A, A_str);
    std::string T_str("/home/spades/kinetic_ws/src/cgmapping/0_non_code_stuff/datasets/matrices_to_test/block_test/T_5x5.matrix");
    cuLiNA::culina_load_matrix_file<double>(T_answer, T_str);

    cuLiNA::cuLiNA_error_t stat;

    stat = cuLiNA::culina_Dblock_assignment_operation(&A, &T, 0, 0, 0, 0, A._getRows(), A._getColumns(), culiopD);

    cuLiNA::cuLiNACheckErrors(stat, __FILE__, __FUNCTION__, __LINE__);

    EXPECT_TRUE( T == T_answer);

}

TEST(block_check, block2x3in3x8start0x3_test){

    cuLiNA::culina_matrix<double, 2, 3> A;
    cuLiNA::culina_matrix<double, 3, 8> T;
    cuLiNA::culina_matrix<double, 3, 8> T_answer;

    cuLiNA::culiopD_t culiopD;

    culiopD.alpha = 1;

    cudaStream_t strm;
    cudaStreamCreate(&strm);

    culiopD.strm = &strm;

    for (int i = 0; i < T._getRows(); ++i) {
        for (int j = 0; j < T._getColumns(); ++j) {

            T(i,j) = 0;

        }
    }

    std::string A_str("/home/spades/kinetic_ws/src/cgmapping/0_non_code_stuff/datasets/matrices_to_test/block_test/A_2x3.matrix");
    cuLiNA::culina_load_matrix_file<double>(A, A_str);
    std::string T_str("/home/spades/kinetic_ws/src/cgmapping/0_non_code_stuff/datasets/matrices_to_test/block_test/T_3x8.matrix");
    cuLiNA::culina_load_matrix_file<double>(T_answer, T_str);

    cuLiNA::cuLiNA_error_t stat;

    stat = cuLiNA::culina_Dblock_assignment_operation(&A, &T, 0, 0, 0, 3, A._getRows(), A._getColumns(), culiopD);

    cuLiNA::cuLiNACheckErrors(stat, __FILE__, __FUNCTION__, __LINE__);

    EXPECT_TRUE( T == T_answer);

}

TEST(block_check, block3x3in7x7start2x2_test){

    cuLiNA::culina_matrix<double, 3, 3> A;
    cuLiNA::culina_matrix<double, 7, 7> T;
    cuLiNA::culina_matrix<double, 7, 7> T_answer;

    cuLiNA::culiopD_t culiopD;

    culiopD.alpha = 1;

    cudaStream_t strm;
    cudaStreamCreate(&strm);

    culiopD.strm = &strm;

    for (int i = 0; i < T._getRows(); ++i) {
        for (int j = 0; j < T._getColumns(); ++j) {

            T(i,j) = 0;

        }
    }

    std::string A_str("/home/spades/kinetic_ws/src/cgmapping/0_non_code_stuff/datasets/matrices_to_test/block_test/A_3x3.matrix");
    cuLiNA::culina_load_matrix_file<double>(A, A_str);
    std::string T_str("/home/spades/kinetic_ws/src/cgmapping/0_non_code_stuff/datasets/matrices_to_test/block_test/T_7x7.matrix");
    cuLiNA::culina_load_matrix_file<double>(T_answer, T_str);

    cuLiNA::cuLiNA_error_t stat;

    stat = cuLiNA::culina_Dblock_assignment_operation(&A, &T, 0, 0, 2, 2, A._getRows(), A._getColumns(), culiopD);

    cudaDeviceSynchronize();
    
    A._printMatrix();
    
    std::cout << std::endl;
    
    T_answer._printMatrix();
    
    std::cout << std::endl;
    
    T._printMatrix();
    
    cuLiNA::cuLiNACheckErrors(stat, __FILE__, __FUNCTION__, __LINE__);

    EXPECT_TRUE( T == T_answer);

}

TEST(block_check, block3x1from3x3in14x3start6x1_test){

    cuLiNA::culina_matrix<double, 3, 3> A;
    cuLiNA::culina_matrix<double, 14, 3> T;
    cuLiNA::culina_matrix<double, 14, 3> T_answer;

    

    for (int i = 0; i < T._getRows(); ++i) {
        for (int j = 0; j < T._getColumns(); ++j) {

            T(i,j) = 0;

        }
    }

    std::string A_str("/home/spades/kinetic_ws/src/cgmapping/0_non_code_stuff/datasets/matrices_to_test/block_test/A_3x3.matrix");
    cuLiNA::culina_load_matrix_file<double>(A, A_str);
    std::string T_str("/home/spades/kinetic_ws/src/cgmapping/0_non_code_stuff/datasets/matrices_to_test/block_test/T_14x3.matrix");
    cuLiNA::culina_load_matrix_file<double>(T_answer, T_str);

//    A._printMatrix();
//
//    T_answer._printMatrix();
    
    cuLiNA::cuLiNA_error_t stat;
    
    cuLiNA::culiopD_t culiopD;
    
    cudaStream_t strm;
    cudaStreamCreate(&strm);
    
    culiopD.strm = &strm;
    culiopD.alpha = 1;
    culiopD.op_m1 = CUBLAS_OP_T;
    
    stat = cuLiNA::culina_Dblock_assignment_operation(&A, &T, 1, 0, 6, 1, 1, A._getColumns(), culiopD);
    
    cudaDeviceSynchronize();
    
    cuLiNA::cuLiNACheckErrors(stat, __FILE__, __FUNCTION__, __LINE__);

    EXPECT_TRUE( T == T_answer);

}

TEST(block_check, block3x3in7x7start6x6_test_fail){

    cuLiNA::culina_matrix<double, 3, 3> A;
    cuLiNA::culina_matrix<double, 7, 7> T;
    cuLiNA::culina_matrix<double, 7, 7> T_answer;

    cuLiNA::culiopD_t culiopD;

    culiopD.alpha = 1;

    cudaStream_t strm;
    cudaStreamCreate(&strm);

    culiopD.strm = &strm;

    for (int i = 0; i < T._getRows(); ++i) {
        for (int j = 0; j < T._getColumns(); ++j) {

            T(i,j) = 0;

        }
    }

    std::string A_str("/home/spades/kinetic_ws/src/cgmapping/0_non_code_stuff/datasets/matrices_to_test/block_test/A_3x3.matrix");
    cuLiNA::culina_load_matrix_file<double>(A, A_str);
    std::string T_str("/home/spades/kinetic_ws/src/cgmapping/0_non_code_stuff/datasets/matrices_to_test/block_test/T_7x7.matrix");
    cuLiNA::culina_load_matrix_file<double>(T_answer, T_str);

    cuLiNA::cuLiNA_error_t stat;

    stat = cuLiNA::culina_Dblock_assignment_operation(&A, &T, 0, 0, 6, 6, A._getRows(), A._getColumns(), culiopD);

    cuLiNA::cuLiNACheckErrors(stat, __FILE__, __FUNCTION__, __LINE__);

    EXPECT_FALSE( T == T_answer);

}

TEST(multiplication_check, multiplication_sqrd_identity_no_acc){

    cuLiNA::culina_matrix4d mat1(cuLiNA::IDENTITY), mat2(cuLiNA::IDENTITY);

    cuLiNA::culina_matrix4d result;

    for (int i = 0; i < result._getRows(); ++i) {
        for (int j = 0; j < result._getColumns(); ++j) {

            result(i,j) = 0;

        }
    }

    cuLiNA::culiopD_t culiopD;

    culiopD.alpha = 1;

    cudaStream_t strm;
    cudaStreamCreate(&strm);

    cuLiNA::cuLiNA_error_t stat;

    stat = cuLiNA::culina_matrix_Dmultiplication(&mat1, &mat2, &result, culiopD);

    cuLiNA::cuLiNACheckErrors(stat, __FILE__, __FUNCTION__, __LINE__);

    EXPECT_TRUE(result == mat1);

}

TEST(multiplication_check, multiplication_sqrd_identity_with_acc){

    cuLiNA::culina_matrix4d mat1(cuLiNA::IDENTITY), mat2(cuLiNA::IDENTITY);

    cuLiNA::culina_matrix4d result, expected_result_matrix;

    for (int i = 0; i < result._getRows(); ++i) {
        for (int j = 0; j < result._getColumns(); ++j) {

            if(i == j)
                expected_result_matrix(i,j) = 2;
            else
                expected_result_matrix(i,j) = 1;

            result(i,j) = 1;

        }
    }

    cuLiNA::culiopD_t culiopD;

    culiopD.alpha = 1;
    culiopD.beta = 1;

    cudaStream_t strm;
    cudaStreamCreate(&strm);

    cuLiNA::cuLiNA_error_t stat;

    stat = cuLiNA::culina_matrix_Dmultiplication(&mat1, &mat2, &result, culiopD);

    cuLiNA::cuLiNACheckErrors(stat, __FILE__, __FUNCTION__, __LINE__);

    EXPECT_TRUE(result == expected_result_matrix);

}

TEST(multiplication_check, diagonal_multiplication_sqrd_identity_with_acc){

    cuLiNA::culina_matrix4d mat1(cuLiNA::IDENTITY), mat2(cuLiNA::DIAGONAL);

    cuLiNA::culina_matrix4d result, expected_result_matrix;

    for (int i = 0; i < result._getRows(); ++i) {
        for (int j = 0; j < result._getColumns(); ++j) {

            if(i == j) {

                expected_result_matrix(i, j) = 2;
                mat2(i,j) = 2;

            } else {
                expected_result_matrix(i, j) = 0;
            }
            result(i,j) = 0;

        }
    }

    cuLiNA::culiopD_t culiopD;

    culiopD.alpha = 1;
    culiopD.beta = 1;

    cudaStream_t strm;
    cudaStreamCreate(&strm);

    cuLiNA::cuLiNA_error_t stat;

    stat = cuLiNA::culina_matrix_Dmultiplication(&mat1, &mat2, &result, culiopD);

    cuLiNA::cuLiNACheckErrors(stat, __FILE__, __FUNCTION__, __LINE__);

    EXPECT_TRUE(result == expected_result_matrix);

}

TEST(multiplication_check, diagonal_multiplication_vector_test){
    
    cuLiNA::culina_matrix4d diag_mat(cuLiNA::DIAGONAL);
    cuLiNA::culina_matrix<double, 4, 1> vec_test;
    cuLiNA::culina_matrix<double, 1, 4> vec_answer, vec_result;
    
    for (int i = 0; i < diag_mat._getRows(); ++i) {
        
        diag_mat(i,i) = 2;
        vec_test(i,0) = 1;
        vec_answer(0,i) = 2;
        
    }
    
//    vec_test._printMatrix();
//    vec_answer._printMatrix();
//    vec_result._printMatrix();
    
    cuLiNA::culiopD_t culiopD;
    
    culiopD.alpha = 1;
    culiopD.beta = 0;
    culiopD.op_m1 = CUBLAS_OP_T;
    
    cudaStream_t strm;
    cudaStreamCreate(&strm);
    
    cuLiNA::cuLiNA_error_t stat;
    
    stat = cuLiNA::culina_matrix_Dmultiplication(&vec_test, &diag_mat, &vec_result, culiopD);
    
    cuLiNA::cuLiNACheckErrors(stat, __FILE__, __FUNCTION__, __LINE__);
    
    //Svec_result._printMatrix();
    
    EXPECT_TRUE(vec_result == vec_answer);
    
}

TEST(multiplication_check, multiplication_diff_sized_matrices7x3by3x7){

    cuLiNA::culina_matrix<double, 3, 7> A;
    cuLiNA::culina_matrix<double, 3, 7> B;
    cuLiNA::culina_matrix<double, 7, 7> C;
    cuLiNA::culina_matrix<double, 7, 7> C_from_matlab;

    cuLiNA::culiopD_t culiopD;

    culiopD.alpha = 1;
    culiopD.beta = 0;
    culiopD.op_m1 = CUBLAS_OP_T;


    cudaStream_t strm;
    cudaStreamCreate(&strm);

    culiopD.strm = &strm;

    std::string A_str("/home/spades/kinetic_ws/src/cgmapping/0_non_code_stuff/datasets/matrices_to_test/mult_test/A_3x7.matrix");
    cuLiNA::culina_load_matrix_file<double>(A, A_str);
    std::string B_str("/home/spades/kinetic_ws/src/cgmapping/0_non_code_stuff/datasets/matrices_to_test/mult_test/B_3x7.matrix");
    cuLiNA::culina_load_matrix_file<double>(B, B_str);
    std::string C_str("/home/spades/kinetic_ws/src/cgmapping/0_non_code_stuff/datasets/matrices_to_test/mult_test/C_7x7.matrix");
    cuLiNA::culina_load_matrix_file<double>(C_from_matlab, C_str);

    //A._printMatrix(true, true);

    cuLiNA::cuLiNA_error_t stat;

    stat = cuLiNA::culina_matrix_Dmultiplication(&A, &B, &C, culiopD);

    cuLiNA::cuLiNACheckErrors(stat, __FILE__, __FUNCTION__, __LINE__);

    cudaDeviceSynchronize();

    EXPECT_TRUE(C == C_from_matlab);


}

TEST(inversion_check, inverse_17x17){

    cuLiNA::culina_matrix<double, 17, 17> A;
    cuLiNA::culina_matrix<double, 17, 17> A_inv;
    cuLiNA::culina_matrix<double, 17, 17> A_inv_from_matlab;
    cuLiNA::culina_matrix<double, 17, 17> mult_result, identity17(cuLiNA::IDENTITY);

    cuLiNA::culiopD_t culiopD;

    culiopD.alpha = 1;
    culiopD.beta = 0;

    cudaStream_t strm;
    cudaStreamCreate(&strm);

    culiopD.strm = &strm;

    cudaMalloc(&culiopD.dev_info, sizeof(int));

    cudaDeviceSynchronize();

    std::string A_str("/home/spades/kinetic_ws/src/cgmapping/0_non_code_stuff/datasets/matrices_to_test/inv_test/A_17x17.matrix");
    cuLiNA::culina_load_matrix_file<double>(A, A_str);
    std::string A_inv_str("/home/spades/kinetic_ws/src/cgmapping/0_non_code_stuff/datasets/matrices_to_test/inv_test/A_inv_17x17.matrix");
    cuLiNA::culina_load_matrix_file<double>(A_inv_from_matlab, A_inv_str);

    //A_inv_from_matlab._printMatrix(true, true);

    cuLiNA::cuLiNA_error_t stat;

    stat = cuLiNA::culina_Dinverse_matrix(&A, &A_inv, culiopD);

    cudaDeviceSynchronize();

    //A_inv._printMatrix();

    cuLiNA::cuLiNACheckErrors(stat, __FILE__, __FUNCTION__, __LINE__);



    stat = cuLiNA::culina_matrix_Dmultiplication(&A, &A_inv, &mult_result, culiopD);
    cuLiNA::cuLiNACheckErrors(stat, __FILE__, __FUNCTION__, __LINE__);


    cudaFree(culiopD.dev_info);

    EXPECT_TRUE((A_inv == A_inv_from_matlab) && (mult_result == identity17));

}

TEST(inversion_check, inverse_17x17_to_same_matrix){

    cuLiNA::culina_matrix<double, 17, 17> A;
    cuLiNA::culina_matrix<double, 17, 17> A_inv_from_matlab;

    cuLiNA::culiopD_t culiopD;

    cudaStream_t strm;
    cudaStreamCreate(&strm);

    culiopD.strm = &strm;

    cudaMalloc(&culiopD.dev_info, sizeof(int));

    std::string A_str("/home/spades/kinetic_ws/src/cgmapping/0_non_code_stuff/datasets/matrices_to_test/inv_test/A_17x17.matrix");
    cuLiNA::culina_load_matrix_file<double>(A, A_str);
    std::string A_inv_str("/home/spades/kinetic_ws/src/cgmapping/0_non_code_stuff/datasets/matrices_to_test/inv_test/A_inv_17x17.matrix");
    cuLiNA::culina_load_matrix_file<double>(A_inv_from_matlab, A_inv_str);

    //A_inv_from_matlab._printMatrix(true, true);

    cuLiNA::cuLiNA_error_t stat;
    stat = cuLiNA::culina_Dinverse_matrix(&A, &A, culiopD);

    cuLiNA::cuLiNACheckErrors(stat, __FILE__, __FUNCTION__, __LINE__);


    cudaFree(culiopD.dev_info);

    EXPECT_TRUE((A == A_inv_from_matlab));

}

TEST(system_solver_check, system_6var_3072data){
    
    cuLiNA::culina_matrix<double, 3072, 6> J;
    cuLiNA::culina_matrix<double, 3072, 3072> W;
    cuLiNA::culina_matrix<double, 3072, 1> data;
    cuLiNA::culina_matrix<double, 6, 1> delta, delta_from_matlab;
    
    cuLiNA::culiopD_t culiopD_1, culiopD_2, culiopD_3;

    cudaStream_t strm[3];
    cudaStreamCreateWithFlags(&strm[0], cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&strm[1], cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&strm[2], cudaStreamNonBlocking);
    

    culiopD_1.strm = &strm[0];
    culiopD_2.strm = &strm[1];
    culiopD_3.strm = &strm[2];

    culiopD_1.workspace = new cuLiNA::culina_matrix<double, 3072, 6>();
    culiopD_2.workspace = new cuLiNA::culina_matrix<double, 6, 1>();
    culiopD_3.workspace = new cuLiNA::culina_matrix<double, 6>();

    std::string J_str("/home/spades/kinetic_ws/src/cgmapping/0_non_code_stuff/datasets/matrices_to_test/linear_solver_test/J_3072x6.matrix");
    cuLiNA::culina_load_matrix_file<double>(J, J_str);
    std::string W_str("/home/spades/kinetic_ws/src/cgmapping/0_non_code_stuff/datasets/matrices_to_test/linear_solver_test/W_3072x3072.matrix");
    cuLiNA::culina_load_matrix_file<double>(W, W_str);
    std::string data_str("/home/spades/kinetic_ws/src/cgmapping/0_non_code_stuff/datasets/matrices_to_test/linear_solver_test/data_3072x1.matrix");
    cuLiNA::culina_load_matrix_file<double>(data, data_str);
    std::string delta_str("/home/spades/kinetic_ws/src/cgmapping/0_non_code_stuff/datasets/matrices_to_test/linear_solver_test/delta_6x1.matrix");
    cuLiNA::culina_load_matrix_file<double>(delta_from_matlab, delta_str);

    std::cout << "before operation" << std::endl;

//    delta._printMatrix();
//    delta_from_matlab._printMatrix();

    
    cuLiNA::cuLiNA_error_t stat;
    stat = cuLiNA::culina_Dsolve_gradient_descent_first_order(&J,
                                                              &delta,
                                                              &data,
                                                              &W,
                                                              0,
                                                              culiopD_1,
                                                              culiopD_2,
                                                              culiopD_3);
    cuLiNA::cuLiNACheckErrors(stat, __FILE__, __FUNCTION__, __LINE__);

    cudaDeviceSynchronize();
    
        EXPECT_TRUE(delta == delta_from_matlab);

    
}

TEST(lie_algebra_check, lie_algebra_exp_map){

    cuLiNA::culina_matrix4d d_homogeneous_matrix, d_homogeneous_matrix_from_matlab;
    cuLiNA::culina_vector3d d_linear_vel, d_angular_vel;
    
    std::string d_linear_vel_str("/home/spades/kinetic_ws/src/cgmapping/0_non_code_stuff/datasets/matrices_to_test/lie_algebra_test/linear_vel_3x1.matrix");
    cuLiNA::culina_load_matrix_file<double>(d_linear_vel, d_linear_vel_str);
    std::string d_angular_vel_str("/home/spades/kinetic_ws/src/cgmapping/0_non_code_stuff/datasets/matrices_to_test/lie_algebra_test/angular_vel_3x1.matrix");
    cuLiNA::culina_load_matrix_file<double>(d_angular_vel, d_angular_vel_str);
    std::string d_homogeneous_matrix_str("/home/spades/kinetic_ws/src/cgmapping/0_non_code_stuff/datasets/matrices_to_test/lie_algebra_test/homogeneous_matrix_4x4.matrix");
    cuLiNA::culina_load_matrix_file<double>(d_homogeneous_matrix_from_matlab, d_homogeneous_matrix_str);
    
    cuLiNA::culiopD_t culiopD_1, culiopD_2, culiopD_3;
    
    //d_angular_vel._setZero();
    
    d_homogeneous_matrix._setZero();
    
    cudaDeviceSynchronize();
    
    cudaStream_t strm[2];
    cudaStreamCreateWithFlags(&strm[0], cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&strm[1], cudaStreamNonBlocking);
    
    culiopD_1.strm = &strm[0];
    culiopD_2.strm = &strm[1];
    
    culiopD_1.workspace = new cuLiNA::culina_matrix3d();
    culiopD_2.workspace = new cuLiNA::culina_matrix3d();
    
    auto stat = cgmapping::cuda::exponential_Dmap_se3(d_linear_vel,
                                                      d_angular_vel,
                                                      d_homogeneous_matrix,
                                                      culiopD_1,
                                                      culiopD_2,
                                                      1.);
    cuLiNA::cuLiNACheckErrors(stat, __FILE__, __FUNCTION__, __LINE__);
    
    cudaDeviceSynchronize();
    
    d_homogeneous_matrix._printMatrix();
    
    d_homogeneous_matrix_from_matlab._printMatrix();
    
    EXPECT_TRUE(d_homogeneous_matrix == d_homogeneous_matrix_from_matlab);
    
}

TEST(lie_algebra_check, lie_algebra_log_map){
    
    cuLiNA::culina_matrix4d d_homogeneous_matrix;
    cuLiNA::culina_vector3d d_linear_vel, d_angular_vel;
    cuLiNA::culina_vector3d d_linear_vel_from_matlab, d_angular_vel_from_matlab;
    
    std::string d_linear_vel_str("/home/spades/kinetic_ws/src/cgmapping/0_non_code_stuff/datasets/matrices_to_test/lie_algebra_test/linear_vel_3x1.matrix");
    cuLiNA::culina_load_matrix_file<double>(d_linear_vel_from_matlab, d_linear_vel_str);
    std::string d_angular_vel_str("/home/spades/kinetic_ws/src/cgmapping/0_non_code_stuff/datasets/matrices_to_test/lie_algebra_test/angular_vel_3x1.matrix");
    cuLiNA::culina_load_matrix_file<double>(d_angular_vel_from_matlab, d_angular_vel_str);
    std::string d_homogeneous_matrix_str("/home/spades/kinetic_ws/src/cgmapping/0_non_code_stuff/datasets/matrices_to_test/lie_algebra_test/homogeneous_matrix_4x4.matrix");
    cuLiNA::culina_load_matrix_file<double>(d_homogeneous_matrix, d_homogeneous_matrix_str);
    
    cuLiNA::culiopD_t culiopD_1, culiopD_2, culiopD_3;
    
    cudaStream_t strm;
    cudaStreamCreateWithFlags(&strm, cudaStreamNonBlocking);
    
    culiopD_1.strm = &strm;
    
    culiopD_1.workspace = new cuLiNA::culina_matrix3d();
   
    Eigen::Vector3d h_linear_vel, h_angular_vel;
    
    Eigen::Matrix4d h_homo_matrix;
    
    cgmapping::Timer tmr;
    auto stat = cgmapping::cuda::logarithmic_Dmap_se3(d_homogeneous_matrix,
                                                      d_linear_vel,
                                                      d_angular_vel,
                                                      culiopD_1,
                                                      1.);
    cuLiNA::cuLiNACheckErrors(stat, __FILE__, __FUNCTION__, __LINE__);
    auto t = tmr.elapsed_in_nsec();
    
    cudaDeviceSynchronize();
    
    d_homogeneous_matrix._printMatrix();
    
    d_homogeneous_matrix._downloadData(h_homo_matrix.data(), 16);
    
    cudaDeviceSynchronize();
    
    std::cout << h_homo_matrix << std::endl;
    
    cgmapping::logarithmic_Dmap_se3(h_homo_matrix, h_linear_vel, h_angular_vel, 1.);
    
    std::cout << h_linear_vel << std::endl;
    std::cout << h_angular_vel << std::endl << std::endl;
    
    d_linear_vel._downloadData(h_linear_vel.data(), 3);
    d_angular_vel._downloadData(h_angular_vel.data(), 3);
    
    cudaDeviceSynchronize();
    
    std::cout << h_linear_vel << std::endl;
    std::cout << h_angular_vel << std::endl << std::endl;
    
    std::cout << "Duration " << t << " [us]" << std::endl;
    
    EXPECT_TRUE(d_linear_vel == d_linear_vel_from_matlab && d_angular_vel == d_angular_vel_from_matlab);
    
}

TEST(sobel_filter_check, sobel_filter_test){
    
    using namespace cv;
    
    cv::Mat img_rgb_t_minus_1, h_abs_rgb_x, h_abs_rgb_y;
    
    img_rgb_t_minus_1 = imread(
        "/home/spades/kinetic_ws/src/cgmapping/0_non_code_stuff/datasets/sudoku-original.jpg",
        CV_LOAD_IMAGE_GRAYSCALE);
    
    cv::cuda::GpuMat d_rgb_t_minus_1_derivative_x, d_img_rgb_t_minus_1, d_abs_rgb_x;
    cv::cuda::GpuMat d_rgb_t_minus_1_derivative_y, d_abs_rgb_y;
    
    cuda::Stream strm;
    cuda::Stream strm2;
    
    d_img_rgb_t_minus_1.upload(img_rgb_t_minus_1, strm);
    
    cv::Mat x_derivative_kernel(1,3,CV_32FC1);
    cv::Mat y_derivative_kernel(3,1,CV_32FC1);
    
    cv::Mat x_inert_kernel(1,3,CV_32FC1);
    cv::Mat y_inert_kernel(3,1,CV_32FC1);
    
    for (int i = 0; i < x_inert_kernel.cols; ++i)
        x_inert_kernel.at<float>(0,i) = 0;
    
    for (int i = 0; i < y_inert_kernel.rows; ++i)
        y_inert_kernel.at<float>(i,0) = 0;
    
    x_inert_kernel.at<float>(0,1) = 1;
    y_inert_kernel.at<float>(1,0) = 1;
    
    x_derivative_kernel.at<float>(0,0) = (float)-0.5;
    y_derivative_kernel.at<float>(0,0) = (float)-0.5;
    x_derivative_kernel.at<float>(0,1) = 0;
    y_derivative_kernel.at<float>(1,0) = 0;
    x_derivative_kernel.at<float>(0,2) = (float)0.5;
    y_derivative_kernel.at<float>(2,0) = (float)0.5;
    
    auto filter_x = cv::cuda::createSeparableLinearFilter(CV_8UC1, CV_16SC1, x_derivative_kernel, y_inert_kernel);
    auto filter_y = cv::cuda::createSeparableLinearFilter(CV_8UC1, CV_16SC1, x_inert_kernel, y_derivative_kernel);
    
//    Ptr<cv::cuda::Filter> filter_x_ = cuda::createSobelFilter(d_img_rgb_t_minus_1.type(), CV_16S, 1, 0, 3, 1, BORDER_DEFAULT);
//    auto filter_y_ = cuda::createSobelFilter(d_img_rgb_t_minus_1.type(), CV_16S, 0, 1, 3, 1, BORDER_DEFAULT);
    //Ptr<cv::cuda::Filter> scharr_filter = cuda::createScharrFilter(d_img_rgb_t_minus_1.type(), CV_16S, 1, 0);
    //auto scharr_filter2 = cuda::createScharrFilter(d_img_rgb_t_minus_1.type(), CV_16S, 0, 1);
    
    
    filter_x->apply(d_img_rgb_t_minus_1, d_rgb_t_minus_1_derivative_x, strm);
    filter_y->apply(d_img_rgb_t_minus_1, d_rgb_t_minus_1_derivative_y, strm2);
    
    //scharr_filter->apply(d_img_rgb_t_minus_1, d_rgb_t_minus_1_derivative_x, strm);
    //scharr_filter2->apply(d_img_rgb_t_minus_1, d_rgb_t_minus_1_derivative_y, strm2);
    
    cv::cuda::abs(d_rgb_t_minus_1_derivative_x, d_rgb_t_minus_1_derivative_x, strm);
    cv::cuda::abs(d_rgb_t_minus_1_derivative_y, d_rgb_t_minus_1_derivative_y, strm2);
    
    d_rgb_t_minus_1_derivative_x.convertTo(d_abs_rgb_x, CV_8UC1, strm);
    d_rgb_t_minus_1_derivative_y.convertTo(d_abs_rgb_y, CV_8UC1, strm2);
    
    d_abs_rgb_x.download(h_abs_rgb_x);
    d_abs_rgb_y.download(h_abs_rgb_y);
    
    cv::namedWindow("Sobel x", CV_8UC1);
    
    imshow("Sobel x", h_abs_rgb_x);
    
    cv::namedWindow("Sobel y", CV_8UC1);
    
    imshow("Sobel y", h_abs_rgb_y);

    while (true) {

        int c;
        c = waitKey(10);

        if ((char) c == 27) break;

    }

    destroyAllWindows();
    
    EXPECT_TRUE(true);

}

TEST(warping_residual_function_check, warping_residual_function_test){
    
    using namespace cv;
    using namespace cv::cuda;
    
    cv::Mat h_img1, h_img2, h_depth_img1, h_img2_warped(480, 640, CV_16UC1), h_img2_warped_filter;
    
    cgmapping::rgb_d_camera_model camera_model(525.0, 525.0, 319.5, 239.5, 5000);
    
    cuLiNA::culina_matrix<double, 480*640, 1> residuals;
    
    h_img1 = cv::imread("/home/spades/kinetic_ws/src/cgmapping/0_non_code_stuff/datasets/rgbd_dataset_freiburg1_desk/rgb/1305031453.359684.png", cv::IMREAD_GRAYSCALE);
    h_img2 = cv::imread("/home/spades/kinetic_ws/src/cgmapping/0_non_code_stuff/datasets/rgbd_dataset_freiburg1_desk/rgb/1305031453.391690.png", cv::IMREAD_GRAYSCALE);
    
    h_depth_img1 = cv::imread("/home/spades/kinetic_ws/src/cgmapping/0_non_code_stuff/datasets/rgbd_dataset_freiburg1_desk/depth/1305031453.374112.png" , CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);
    
    cv::cuda::Stream strm[3];
    
    cgmapping::image_size_t IMG_SIZE_USED = cgmapping::ORIGINAL_SIZE;
    
    cgmapping::image_pyramid d_img1_pyr(480, 640, CV_8U);
    cgmapping::image_pyramid d_img2_pyr(480, 640, CV_8U);
    cgmapping::image_pyramid d_img2_warped_pyr(480, 640, CV_8U);
    
    cgmapping::image_pyramid d_depth1_pyr(480, 640, CV_16U);
    
    //this one is used to filter usable pixels
    cgmapping::image_pyramid d_filter_warped_pyr(480, 640, CV_8U);
    
    d_img1_pyr._generate_pyramid(h_img1, strm[0]);
    d_img2_pyr._generate_pyramid(h_img2, strm[1]);
    d_depth1_pyr._generate_pyramid(h_depth_img1, strm[2]);
    
    cuLiNA::culina_matrix4d homogenic_transformation;
    cuLiNA::culina_matrix<double, 480*640, 3> d_img1_oct_point_cloud;
    
    int rows = d_depth1_pyr._getImageMat(IMG_SIZE_USED).rows;
    int cols = d_depth1_pyr._getImageMat(IMG_SIZE_USED).cols;
    
    d_img1_oct_point_cloud._setRows(rows*cols);
    
    homogenic_transformation._setIdentity();
    
    cudaDeviceSynchronize();
    
    /////////////////////////////////////////WARPING IMAGE////////////////////////////////////////////
    
    cgmapping::cuda::calculate_Dimage_warped(d_img2_pyr._getImageMat(IMG_SIZE_USED),
                                             d_img2_warped_pyr._getImageMat(IMG_SIZE_USED),
                                             d_filter_warped_pyr._getImageMat(IMG_SIZE_USED),
                                             d_depth1_pyr._getImageMat(IMG_SIZE_USED),
                                             d_img1_oct_point_cloud,
                                             homogenic_transformation,
                                             camera_model,
                                             strm[0]);
    
    cudaDeviceSynchronize();
    
    //d_img1_oct_point_cloud._printMatrix();
    
    d_img2_warped_pyr._getImageMat(IMG_SIZE_USED).download(h_img2_warped, strm[0]);
    d_filter_warped_pyr._getImageMat(IMG_SIZE_USED).download(h_img2_warped_filter, strm[0]);

    residuals._setRows(rows);
    residuals._setColumns(cols);
    
    /////////////////////////////////////////IMAGE RESIDUAL////////////////////////////////////////////
    
    cgmapping::cuda::calculate_Dimage_residual(d_img1_pyr._getImageMat(IMG_SIZE_USED),
                                               d_img2_warped_pyr._getImageMat(IMG_SIZE_USED),
                                               d_filter_warped_pyr._getImageMat(IMG_SIZE_USED),
                                               residuals,
                                               strm[0]);
    
    cudaDeviceSynchronize();
    
    cuLiNA::culina_matrix<double, 640*480*2, 6> warp_jacobian;
    cuLiNA::culina_matrix<double, 640*480, 6> full_jacobian;
    
    warp_jacobian._setRows(rows*cols*2);
    full_jacobian._setRows(rows*cols);
    
    cgmapping::image_pyramid img_warped_derivative_x(480, 640, CV_16S);
    cgmapping::image_pyramid img_warped_derivative_y(480, 640, CV_16S);
    
    /////////////////////////////////////////WARP IMAGE DERIVATIVES////////////////////////////////////////////
    
    cv::Mat x_derivative_kernel(1,3,CV_32FC1);
    cv::Mat y_derivative_kernel(3,1,CV_32FC1);
    
    cv::Mat x_inert_kernel(1,3,CV_32FC1);
    cv::Mat y_inert_kernel(3,1,CV_32FC1);
    
    for (int i = 0; i < x_inert_kernel.cols; ++i)
        x_inert_kernel.at<float>(0,i) = 0;
    
    for (int i = 0; i < y_inert_kernel.rows; ++i)
        y_inert_kernel.at<float>(i,0) = 0;
    
    x_inert_kernel.at<float>(0,1) = 1;
    y_inert_kernel.at<float>(1,0) = 1;
    
    x_derivative_kernel.at<float>(0,0) = (float)-0.5;
    y_derivative_kernel.at<float>(0,0) = (float)-0.5;
    x_derivative_kernel.at<float>(0,1) = 0;
    y_derivative_kernel.at<float>(1,0) = 0;
    x_derivative_kernel.at<float>(0,2) = (float)0.5;
    y_derivative_kernel.at<float>(2,0) = (float)0.5;
    
    auto filter_x = cv::cuda::createSeparableLinearFilter(CV_8UC1, CV_16SC1, x_derivative_kernel, y_inert_kernel);
    auto filter_y = cv::cuda::createSeparableLinearFilter(CV_8UC1, CV_16SC1, x_inert_kernel, y_derivative_kernel);
    
//    Ptr<cv::cuda::Filter> filter_x = cuda::createSobelFilter(d_img2_warped_pyr._getImageMat(IMG_SIZE_USED).type(),
//                                                                   CV_16S, 1, 0, 3, 1, BORDER_DEFAULT);
//    auto filter_y = cuda::createSobelFilter(d_img2_warped_pyr._getImageMat(IMG_SIZE_USED).type(),
//                                                  CV_16S, 0, 1, 3, 1, BORDER_DEFAULT);
    
    filter_x->apply(d_img2_warped_pyr._getImageMat(IMG_SIZE_USED),
                          img_warped_derivative_x._getImageMat(IMG_SIZE_USED), strm[0]);
    filter_y->apply(d_img2_warped_pyr._getImageMat(IMG_SIZE_USED),
                          img_warped_derivative_y._getImageMat(IMG_SIZE_USED), strm[1]);
    
    
    /////////////////////////////////////////CALCULATING JACOBIAN////////////////////////////////////////////
    
    cgmapping::cuda::calculate_Dwarp_jacobian(warp_jacobian, d_img1_oct_point_cloud, camera_model, strm[2]);

    cudaDeviceSynchronize();

    //warp_jacobian._printMatrix();
    
    cgmapping::cuda::calculate_Dfull_jacobian(img_warped_derivative_x._getImageMat(IMG_SIZE_USED),
                                              img_warped_derivative_y._getImageMat(IMG_SIZE_USED),
                                              warp_jacobian, full_jacobian, strm[0]);

    cudaDeviceSynchronize();
    
    //full_jacobian._printMatrix();
    
    cv::Mat h_residual_image(d_img2_warped_pyr._getImageMat(IMG_SIZE_USED).rows,
                             d_img2_warped_pyr._getImageMat(IMG_SIZE_USED).cols,
                             cv::DataType<char>::type);
    
    cv::Mat h_jacob_v1_image(d_img2_warped_pyr._getImageMat(IMG_SIZE_USED).rows,
                             d_img2_warped_pyr._getImageMat(IMG_SIZE_USED).cols,
                             cv::DataType<short>::type);
    
    cv::Mat h_jacob_v2_image(d_img2_warped_pyr._getImageMat(IMG_SIZE_USED).rows,
                             d_img2_warped_pyr._getImageMat(IMG_SIZE_USED).cols,
                             cv::DataType<short>::type);
    
    cv::Mat h_jacob_v3_image(d_img2_warped_pyr._getImageMat(IMG_SIZE_USED).rows,
                             d_img2_warped_pyr._getImageMat(IMG_SIZE_USED).cols,
                             cv::DataType<short>::type);
    
    cv::Mat h_jacob_w1_image(d_img2_warped_pyr._getImageMat(IMG_SIZE_USED).rows,
                             d_img2_warped_pyr._getImageMat(IMG_SIZE_USED).cols,
                             cv::DataType<short>::type);
    
    cv::Mat h_jacob_w2_image(d_img2_warped_pyr._getImageMat(IMG_SIZE_USED).rows,
                             d_img2_warped_pyr._getImageMat(IMG_SIZE_USED).cols,
                             cv::DataType<short>::type);
    
    cv::Mat h_jacob_w3_image(d_img2_warped_pyr._getImageMat(IMG_SIZE_USED).rows,
                             d_img2_warped_pyr._getImageMat(IMG_SIZE_USED).cols,
                             cv::DataType<short>::type);
    
    cv::Mat h_abs_rgb_x, h_abs_rgb_y;
    
    //residuals._printMatrix();
    
    for (int j = 0; j < d_img2_warped_pyr._getImageMat(IMG_SIZE_USED).cols; ++j) {

        for (int i = 0; i < d_img2_warped_pyr._getImageMat(IMG_SIZE_USED).rows; ++i) {

            short value = saturate_cast<short>(residuals._getRawData()[IDX2X_CM(i,j,residuals._getLeading_dimension())]);

            auto jacob_row_idx = IDX2X_CM(i,j,residuals._getLeading_dimension());

            h_residual_image.at<char>(i,j) = (((abs(value)<1000)?value:0));

            h_jacob_v1_image.at<short>(i,j) = full_jacobian._getRawData()[IDX2X_CM(jacob_row_idx, 0,full_jacobian._getLeading_dimension())];
            h_jacob_v2_image.at<short>(i,j) = full_jacobian._getRawData()[IDX2X_CM(jacob_row_idx, 1,full_jacobian._getLeading_dimension())];
            h_jacob_v3_image.at<short>(i,j) = full_jacobian._getRawData()[IDX2X_CM(jacob_row_idx, 2,full_jacobian._getLeading_dimension())];
            h_jacob_w1_image.at<short>(i,j) = full_jacobian._getRawData()[IDX2X_CM(jacob_row_idx, 3,full_jacobian._getLeading_dimension())];
            h_jacob_w2_image.at<short>(i,j) = full_jacobian._getRawData()[IDX2X_CM(jacob_row_idx, 4,full_jacobian._getLeading_dimension())];
            h_jacob_w3_image.at<short>(i,j) = full_jacobian._getRawData()[IDX2X_CM(jacob_row_idx, 5,full_jacobian._getLeading_dimension())];

            //std::cout << h_residual_image.at<char>(i,j) << std::endl;

        }

    }
    
    cv::cuda::abs(img_warped_derivative_x._getImageMat(IMG_SIZE_USED),
                  img_warped_derivative_x._getImageMat(IMG_SIZE_USED),
                  strm[0]);
    cv::cuda::abs(img_warped_derivative_y._getImageMat(IMG_SIZE_USED),
                  img_warped_derivative_y._getImageMat(IMG_SIZE_USED),
                  strm[1]);
    
    //h_jacob_v1_image = cv::abs(h_jacob_v1_image);
    
    cv::cuda::GpuMat d_abs_rgb_x, d_abs_rgb_y;
    
    img_warped_derivative_x._getImageMat(IMG_SIZE_USED).convertTo(d_abs_rgb_x, CV_8UC1, strm[0]);
    img_warped_derivative_y._getImageMat(IMG_SIZE_USED).convertTo(d_abs_rgb_y, CV_8UC1, strm[1]);
    
    d_abs_rgb_x.download(h_abs_rgb_x);
    d_abs_rgb_y.download(h_abs_rgb_y);
    
    h_residual_image.convertTo(h_residual_image, cv::DataType<uchar>::type);
    
    //h_jacob_v1_image.convertTo(h_jacob_v1_image, cv::DataType<uchar>::type);
    
    cv::namedWindow("img 1", cv::IMREAD_GRAYSCALE);

    imshow("img 1", h_img1);

    cv::namedWindow("img 2", cv::IMREAD_GRAYSCALE);

    imshow("img 2", h_img2);
    
    cv::namedWindow("depth img 1", CV_16UC1);

    imshow("depth img 1", h_depth_img1);

    cv::namedWindow("img 2 warped", CV_8UC1);

    imshow("img 2 warped", h_img2_warped);
    
    cv::namedWindow("img 2 warped filter", CV_8U);
    
    imshow("img 2 warped filter", h_img2_warped_filter);

    cv::namedWindow("residual img", cv::DataType<uchar>::type);

    imshow("residual img", h_residual_image);
    
    cv::namedWindow("Sobel x", CV_8UC1);
    
    imshow("Sobel x", h_abs_rgb_x);
    
    cv::namedWindow("Sobel y", CV_8UC1);
    
    imshow("Sobel y", h_abs_rgb_y);
    
    cv::namedWindow("jacobian v1", cv::DataType<short>::type);

    imshow("jacobian v1", h_jacob_v1_image);

    cv::namedWindow("jacobian v2", cv::DataType<short>::type);

    imshow("jacobian v2", h_jacob_v2_image);

    cv::namedWindow("jacobian v3", cv::DataType<short>::type);

    imshow("jacobian v3", h_jacob_v3_image);

    cv::namedWindow("jacobian w1", cv::DataType<short>::type);

    imshow("jacobian w1", h_jacob_w1_image);

    cv::namedWindow("jacobian w2", cv::DataType<short>::type);

    imshow("jacobian w2", h_jacob_w2_image);

    cv::namedWindow("jacobian w3", cv::DataType<short>::type);

    imshow("jacobian w3", h_jacob_w3_image);

    while (true) {

        int c;
        c = waitKey(10);

        if ((char) c == 27) break;

    }

    
    EXPECT_TRUE(true);
    
}

TEST(matrix_assignment_check, matrix_assignment_test){
    
    cuLiNA::culina_matrix<double, 3, 3> matrix_test;
    cuLiNA::culina_matrix<double, 3, 3> matrix_test1;
    
    matrix_test._setIdentity();
    matrix_test1._setZero();
    
    cudaStream_t strm;
    cudaStreamCreateWithFlags(&strm, cudaStreamNonBlocking);
    
    cgmapping::Timer timer;
    //matrix_test1 = static_cast<cuLiNA::culina_tm<double>&>(matrix_test);
    
    matrix_test1._loadData(matrix_test, strm);
    
    //matrix_test = static_cast<cuLiNA::culina_tm<double>&>(matrix_test);
    
    cudaDeviceSynchronize();
    auto tmp = timer.elapsed_in_nsec();
    
    std::cout << "time = " << tmp << std::endl;
    
    EXPECT_TRUE(matrix_test == matrix_test1);
    
}

TEST(weight_matrix_check, weight_matrix_test){
    
    using namespace cv;
    using namespace cv::cuda;
    
    cv::Mat h_img1, h_img2, h_depth_img1, h_img2_warped(480, 640, CV_16UC1), h_img2_warped_filter;
    
    cgmapping::rgb_d_camera_model camera_model(525.0, 525.0, 319.5, 239.5, 5000);
    
    cuLiNA::culina_matrix<double, 480*640, 1> residuals;
    cuLiNA::culina_matrix<double, 480*640, 480*640> weight_matrix(DIAGONAL);
    
    h_img1 = cv::imread("/home/spades/kinetic_ws/src/cgmapping/0_non_code_stuff/datasets/rgbd_dataset_freiburg1_desk/rgb/1305031453.359684.png", cv::IMREAD_GRAYSCALE);
    h_img2 = cv::imread("/home/spades/kinetic_ws/src/cgmapping/0_non_code_stuff/datasets/rgbd_dataset_freiburg1_desk/rgb/1305031453.391690.png", cv::IMREAD_GRAYSCALE);
    
    h_depth_img1 = cv::imread("/home/spades/kinetic_ws/src/cgmapping/0_non_code_stuff/datasets/rgbd_dataset_freiburg1_desk/depth/1305031453.374112.png" , CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);
    
    cv::cuda::Stream strm[3];
    
    cgmapping::image_size_t IMG_SIZE_USED = cgmapping::ORIGINAL_SIZE;
    
    cgmapping::image_pyramid d_img1_pyr(480, 640, CV_8U);
    cgmapping::image_pyramid d_img2_pyr(480, 640, CV_8U);
    cgmapping::image_pyramid d_img2_warped_pyr(480, 640, CV_8U);
    
    cgmapping::image_pyramid d_depth1_pyr(480, 640, CV_16U);
    
    //this one is used to filter usable pixels
    cgmapping::image_pyramid d_filter_warped_pyr(480, 640, CV_8U);
    
    d_img1_pyr._generate_pyramid(h_img1, strm[0]);
    d_img2_pyr._generate_pyramid(h_img2, strm[1]);
    d_depth1_pyr._generate_pyramid(h_depth_img1, strm[2]);
    
    cuLiNA::culina_matrix4d homogenic_transformation;
    cuLiNA::culina_matrix<double, 480*640, 3> d_img1_oct_point_cloud;
    
    int rows = d_depth1_pyr._getImageMat(IMG_SIZE_USED).rows;
    int cols = d_depth1_pyr._getImageMat(IMG_SIZE_USED).cols;
    
    d_img1_oct_point_cloud._setRows(rows*cols);
    
    homogenic_transformation._setIdentity();
    
    cudaDeviceSynchronize();
    
    /////////////////////////////////////////WARPING IMAGE////////////////////////////////////////////
    
    cgmapping::cuda::calculate_Dimage_warped(d_img2_pyr._getImageMat(IMG_SIZE_USED),
                                             d_img2_warped_pyr._getImageMat(IMG_SIZE_USED),
                                             d_filter_warped_pyr._getImageMat(IMG_SIZE_USED),
                                             d_depth1_pyr._getImageMat(IMG_SIZE_USED),
                                             d_img1_oct_point_cloud,
                                             homogenic_transformation,
                                             camera_model,
                                             strm[0]);
    
    cudaDeviceSynchronize();
    
    //d_img1_oct_point_cloud._printMatrix();
    
    d_img2_warped_pyr._getImageMat(IMG_SIZE_USED).download(h_img2_warped, strm[0]);
    d_filter_warped_pyr._getImageMat(IMG_SIZE_USED).download(h_img2_warped_filter, strm[0]);
    
    residuals._setRows(rows);
    residuals._setColumns(cols);
    
    //weight_matrix(rows*cols);
    
    /////////////////////////////////////////IMAGE RESIDUAL////////////////////////////////////////////
    
    cgmapping::cuda::calculate_Dimage_residual(d_img1_pyr._getImageMat(IMG_SIZE_USED),
                                               d_img2_warped_pyr._getImageMat(IMG_SIZE_USED),
                                               d_filter_warped_pyr._getImageMat(IMG_SIZE_USED),
                                               residuals,
                                               strm[0]);
    
    cudaDeviceSynchronize();
    
    cv::Mat h_residual_image(d_img2_warped_pyr._getImageMat(IMG_SIZE_USED).rows,
                             d_img2_warped_pyr._getImageMat(IMG_SIZE_USED).cols,
                             cv::DataType<char>::type);
    
    cv::Mat h_weight_image(d_img2_warped_pyr._getImageMat(IMG_SIZE_USED).rows,
                             d_img2_warped_pyr._getImageMat(IMG_SIZE_USED).cols,
                             cv::DataType<double>::type);
    
    auto number_of_valid_data = cgmapping::cuda::calculate_Dnumber_of_valid_data(residuals, strm[0]);
    
    auto std_dev = cgmapping::cuda::calculate_Dstandard_deviation_t_student(residuals, 5, 1, number_of_valid_data, 0.0000001, strm[0]);
    
    auto variance = std_dev*std_dev;
    
    
    
    weight_matrix._setColumns(rows*cols);
    weight_matrix._setRows(rows*cols);
    
    cgmapping::cuda::calculate_Dweight_matrix(residuals, weight_matrix, 5, variance, strm[0]);
    
    cuLiNA::culina_base_matrix<double> auxiliar_matrix(1, rows*cols, 1);
    cuLiNA::culina_matrix<double, 1, 1> sqrd_weighted_error;
    
    
    //auxiliar_matrix._printMatrix();
    
    cudaDeviceSynchronize();
    
    std::cout << "here0 0" << std::endl;
    
//    for (int k = 0; k < weight_matrix._getRows(); ++k) {
//
//        std::cout << "Weight["<< k << "," << k << "] = " << weight_matrix(k,k) << std::endl;
//
//    }
    
    residuals._setColumns(1);
    residuals._setRows(rows*cols);
    
//    cgmapping::cuda::calculate_Dsquared_weighted_error(residuals,
//                                                       weight_matrix,
//                                                       sqrd_weighted_error_,
//                                                       auxiliar_matrix,
//                                                       number_of_valid_data,
//                                                       strm[0]);
    
//    std::cout << std::endl << "SQRD WEIGHTED ERROR ------------> ";
//    sqrd_weighted_error_._printMatrix();
//    std::cout << std::endl;
    
    residuals._setColumns(cols);
    residuals._setRows(rows);
    
    //residuals._printMatrix();
    
    for (int j = 0; j < d_img2_warped_pyr._getImageMat(IMG_SIZE_USED).cols; ++j) {
        
        for (int i = 0; i < d_img2_warped_pyr._getImageMat(IMG_SIZE_USED).rows; ++i) {
            
            short value = saturate_cast<short>(residuals._getRawData()[IDX2X_CM(i,j,residuals._getLeading_dimension())]);
            
            auto row_idx = IDX2X_CM(i,j,residuals._getLeading_dimension());
            
            h_residual_image.at<char>(i,j) = (((abs(value)<1000)?value:-127));
            h_weight_image.at<double>(i,j) = weight_matrix._getRawData()[row_idx];
            
            //std::cout << h_residual_image.at<char>(i,j) << std::endl;
            
        }
        
    }
    
    //h_weight_image.convertTo(h_weight_image, cv::IMREAD_GRAYSCALE);
    
    cv::namedWindow("img 1", cv::IMREAD_GRAYSCALE);
    
    imshow("img 1", h_img1);
    
    cv::namedWindow("img 2", cv::IMREAD_GRAYSCALE);
    
    imshow("img 2", h_img2);
    
    cv::namedWindow("depth img 1", CV_16UC1);
    
    imshow("depth img 1", h_depth_img1);
    
    cv::namedWindow("img 2 warped", CV_8UC1);
    
    imshow("img 2 warped", h_img2_warped);
    
    cv::namedWindow("img 2 warped filter", CV_8U);
    
    imshow("img 2 warped filter", h_img2_warped_filter);
    
    cv::namedWindow("residual img", cv::DataType<uchar>::type);
    
    imshow("residual img", h_residual_image);
    
    cv::namedWindow("weight img", cv::DataType<double>::type);
    
    imshow("weight img", h_weight_image);
    
    while (true) {
        
        int c;
        c = waitKey(10);
        
        if ((char) c == 27) break;
        
    }
    
    
    EXPECT_TRUE(true);
    
}

TEST(camera_motion_check, camera_motion_test){

    cv::Mat h_img1, h_img2, h_depth_img1, h_img2_warped, h_img2_warped_filter;
    
    cgmapping::rgb_d_camera_model camera_model(525.0, 525.0, 319.5, 239.5, 5000);
    
    h_img1 = cv::imread("/home/spades/kinetic_ws/src/cgmapping/0_non_code_stuff/datasets/rgbd_dataset_freiburg1_desk/rgb/1305031453.359684.png", cv::IMREAD_GRAYSCALE);
    h_img2 = cv::imread("/home/spades/kinetic_ws/src/cgmapping/0_non_code_stuff/datasets/rgbd_dataset_freiburg1_desk/rgb/1305031453.391690.png", cv::IMREAD_GRAYSCALE);
    
    h_depth_img1 = cv::imread("/home/spades/kinetic_ws/src/cgmapping/0_non_code_stuff/datasets/rgbd_dataset_freiburg1_desk/depth/1305031453.374112.png" , CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);
    
    double k_accptance_error = 0.00001;
    uint num_it = 10000;
    
    Eigen::Matrix4d tmp_matrix;
    
    cgmapping::cuda::visual_odometer<double> visual_odom(h_img1, h_depth_img1, camera_model, k_accptance_error, num_it, 5);

    visual_odom._estimateCamera_motion(h_img2, h_depth_img1, cgmapping::HEX_SIZE, cgmapping::ORIGINAL_SIZE);
    
    visual_odom._getCur_estimated_pose()._downloadData(tmp_matrix.data(), 16);
    
    cudaDeviceSynchronize();
    
    std::cout << std::endl << tmp_matrix << std::endl;
    
    std::cout << "num_its = "  << visual_odom._getIterations_at_img_resolution(cgmapping::HEX_SIZE) << std::endl;
    std::cout << "num_its = "  << visual_odom._getIterations_at_img_resolution(cgmapping::OCT_SIZE) << std::endl;
    std::cout << "num_its = "  << visual_odom._getIterations_at_img_resolution(cgmapping::QUARTER_SIZE) << std::endl;
    std::cout << "num_its = "  << visual_odom._getIterations_at_img_resolution(cgmapping::HALF_SIZE) << std::endl;
    std::cout << "num_its = "  << visual_odom._getIterations_at_img_resolution(cgmapping::ORIGINAL_SIZE) << std::endl;
    
    
    EXPECT_TRUE(true);
    

}

TEST(for_enum_check, for_enum_test){
    
    for (int j = cgmapping::HEX_SIZE; j <= cgmapping::ORIGINAL_SIZE ; j++) {
        
        auto img_size = static_cast<cgmapping::image_size_t>(j);
        
        switch(j){
            
            case cgmapping::HEX_SIZE: std::cout << "HEX" << std::endl;
                                      break;
            case cgmapping::OCT_SIZE: std::cout << "OCT" << std::endl;
                break;
            case cgmapping::QUARTER_SIZE: std::cout << "QUARTER" << std::endl;
                break;
            case cgmapping::HALF_SIZE: std::cout << "HALF" << std::endl;
                break;
            case cgmapping::ORIGINAL_SIZE: std::cout << "ORIGINAL" << std::endl;
            defautl: break;
        }
        
    }
    
}

TEST(eigen_matrix_and_culina_matrix_check, eigen_matrix2culina_matrix_test){
    
    cuLiNA::culina_matrix3d d_matrix;
    Eigen::Matrix3d h_matrix;
    
    d_matrix._setDiagonalValue(0.00000000001);
    
    cudaDeviceSynchronize();
    
    d_matrix._printMatrix();
    
    std::string file_name("/home/spades/kinetic_ws/src/cgmapping/0_non_code_stuff/a_matrix.m");
    
    cuLiNA::culina_download_matrix_file(d_matrix, file_name);
    
    h_matrix.setZero(3,3);
    
    std::cout << h_matrix << std::endl;
    
    d_matrix._downloadData(h_matrix.data(), 9);
    
    cudaDeviceSynchronize();
    
    std::cout << h_matrix << std::endl;
    
}

TEST(eigen_matrix_and_culina_matrix_check, culina_matrix2eigen_matrix_test){
    
    cuLiNA::culina_matrix3d d_matrix;
    Eigen::Matrix3d h_matrix;
    
    h_matrix.setRandom(3,3);
    
    std::cout << h_matrix << std::endl;
    
    d_matrix._setZero();
    
    cudaDeviceSynchronize();
    
    d_matrix._printMatrix();
    
    d_matrix._uploadData(h_matrix.data(), 9);
    
    cudaDeviceSynchronize();
    
    d_matrix._printMatrix();
    
}

TEST(cv_mat_rgb_grayscale_gpu_check, cv_mat_rgb_grayscale_gpu_test){
    
    Mat h_img1, h_img2;
    
    h_img1 = cv::imread("/home/spades/kinetic_ws/src/cgmapping/0_non_code_stuff/datasets/rgbd_dataset_freiburg1_desk/rgb/1305031453.359684.png");
    
    cgmapping::image_pyramid d_pyr_img1(h_img1.rows, h_img1.cols, CV_8UC1);
    
    d_pyr_img1._generate_pyramid(h_img1);
    
    cudaDeviceSynchronize();
    
    d_pyr_img1._getImageMat(cgmapping::ORIGINAL_SIZE).download(h_img2);
    
    cudaDeviceSynchronize();
    
//    cv::namedWindow("img 1", h_img1.type());
//
//    imshow("img 1", h_img1);
    
    cv::namedWindow("img 2", h_img2.type());
    
    imshow("img 2", h_img2);
    
    while (true) {
        
        int c;
        c = waitKey(10);
        
        if ((char) c == 27) break;
        
    }
    
}

TEST(visual_odometry_check, visual_odometry_test){

    using namespace std;
    using namespace boost;
    
    fstream rgb_stream, depth_stream, groundtruth_stream;
    
    //string database_path("/home/spades/kinetic_ws/src/cgmapping/0_non_code_stuff/datasets/rgbd_dataset_freiburg1_xyz/");
    string database_path("/home/spades/kinetic_ws/src/cgmapping/0_non_code_stuff/datasets/rgbd_dataset_freiburg1_desk/");
    
    string rgb_file_name = database_path + "rgb.txt";
    string depth_file_name = database_path + "depth.txt";
    string groundtruth_file_name = database_path + "groundtruth.txt";
    
    rgb_stream.open(rgb_file_name);
    if(!rgb_stream.is_open())
        cout << "Not possible to open rgb name list file" << endl;
    
    depth_stream.open(depth_file_name);
    if(!depth_stream.is_open())
        cout << "Not possible to open depth name list file" << endl;
    
    groundtruth_stream.open(groundtruth_file_name);
    if(!groundtruth_stream.is_open())
        cout << "Not possible to open groundtruth file" << endl;
    
    cgmapping::syncronize_fstreams_from_rgbd_data_set(rgb_stream, depth_stream, groundtruth_stream);
    
    std::string pose_estimation_file("pose_estimation.txt");
    
    pose_estimation_file = database_path + pose_estimation_file;
    
    string rgb_image_file_name;
    string depth_image_file_name;
    string groundtruth_init_pos;
    
    vector<string> rgb_vec;
    vector<string> depth_vec;
    vector<string> groundtruth_vec;
    
    getline(rgb_stream, rgb_image_file_name);
    getline(depth_stream, depth_image_file_name);
    getline(groundtruth_stream, groundtruth_init_pos);
    
    split_regex(rgb_vec, rgb_image_file_name, regex(" "));
    split_regex(depth_vec, depth_image_file_name, regex(" "));
    split_regex(groundtruth_vec, groundtruth_init_pos, regex(" "));
    
    rgb_image_file_name = database_path + rgb_vec[1];
    depth_image_file_name = database_path + depth_vec[1];
    
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    cv::Mat h_cur_img, h_cur_depth;
    
    h_cur_img = cv::imread(rgb_image_file_name, cv::IMREAD_GRAYSCALE);
    h_cur_depth = cv::imread(depth_image_file_name, CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);
    
    cgmapping::Timer __timer;
    vector<double> time_stamps;
    
    Eigen::Matrix4d cur_optical_frame_pose;
    geometry_msgs::Pose cur_optical_frame_pose_quaternion;
    
    cgmapping::pose_line_2_eigen_pose_matrix4d(groundtruth_init_pos, cur_optical_frame_pose);
    cgmapping::eigen_pose_matrix4d_2_ros_geometry_msg_pose(cur_optical_frame_pose, cur_optical_frame_pose_quaternion);
    
    vector<geometry_msgs::Pose> optical_rgb_frame_estimated_poses;
    
    cudaStream_t raw_strm[4];
    cv::cuda::Stream strm[4];
    
    for (int i = 0; i < 4; ++i) {
        strm[i] = cv::cuda::StreamAccessor::wrapStream(raw_strm[i]);
    }
    
    cgmapping::rgb_d_camera_model camera_model(525.0, 525.0, 319.5, 239.5, 5000);
    
    double eps_error = 0.001;
    uint max_iterations = 40;
    double t_student_degrees_of_freedom = 5;
    double eps_acceptance_std_dev_t_student = 0.0001;
    
    cgmapping::image_size_t min_size = cgmapping::HEX_SIZE;
    cgmapping::image_size_t max_size = cgmapping::ORIGINAL_SIZE;
    
    cgmapping::cuda::visual_odometer<double> __vo(h_cur_img,
                                                  h_cur_depth,
                                                  camera_model,
                                                  eps_error,
                                                  max_iterations,
                                                  t_student_degrees_of_freedom,
                                                  eps_acceptance_std_dev_t_student,
                                                  cur_optical_frame_pose.data());
    
    time_stamps.push_back(atof(groundtruth_vec[0].c_str()));
    optical_rgb_frame_estimated_poses.push_back(cur_optical_frame_pose_quaternion);
    
//    cv::namedWindow("RGB img", cv::IMREAD_GRAYSCALE);
//    cv::namedWindow("DEPTH img", CV_16UC1);
    
    cout << rgb_image_file_name << endl;
    cout << depth_image_file_name << endl;
    
    int num_estimations = 0;
    float frequency_sum = 0;
    
    while(getline(rgb_stream, rgb_image_file_name) &&  getline(depth_stream, depth_image_file_name)){
        
        if(!rgb_stream.eof() && !depth_stream.eof() && rgb_image_file_name[0] != '#' && depth_image_file_name[0] != '#') {
    
            rgb_vec.clear();
            split_regex(rgb_vec, rgb_image_file_name, regex(" "));
            rgb_image_file_name = database_path + rgb_vec[1];
    
            depth_vec.clear();
            split_regex(depth_vec, depth_image_file_name, regex(" "));
            depth_image_file_name = database_path + depth_vec[1];
    
            
            
            h_cur_img = cv::imread(rgb_image_file_name, cv::IMREAD_GRAYSCALE);
            h_cur_depth = cv::imread(depth_image_file_name, CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);
    
            __timer.reset();
            
            __vo._estimateCamera_motion(h_cur_img, h_cur_depth, min_size, max_size);
            
            auto time_elapsed = __timer.elapsed_in_sec();
            
            frequency_sum += 1./time_elapsed;
            num_estimations++;
            
            __vo._getCur_estimated_pose()._downloadData(cur_optical_frame_pose.data(), 16, &raw_strm[0]);

            strm[0].waitForCompletion();
    
            std::cout << cur_optical_frame_pose << std::endl;

            std::cout << std::endl;
            std::cout << "avarage frequency = " << frequency_sum/num_estimations << std::endl;
            std::cout << "time elapsed = " << time_elapsed << std::endl;
            std::cout << "instant frequency = " << 1./time_elapsed << std::endl;
            std::cout << "num_its = "  << __vo._getIterations_at_img_resolution(cgmapping::HEX_SIZE) << std::endl;
            std::cout << "num_its = "  << __vo._getIterations_at_img_resolution(cgmapping::OCT_SIZE) << std::endl;
            std::cout << "num_its = "  << __vo._getIterations_at_img_resolution(cgmapping::QUARTER_SIZE) << std::endl;
            std::cout << "num_its = "  << __vo._getIterations_at_img_resolution(cgmapping::HALF_SIZE) << std::endl;
            std::cout << "num_its = "  << __vo._getIterations_at_img_resolution(cgmapping::ORIGINAL_SIZE) << std::endl;
            std::cout << std::endl;
            
            cgmapping::eigen_pose_matrix4d_2_ros_geometry_msg_pose(cur_optical_frame_pose, cur_optical_frame_pose_quaternion);

            optical_rgb_frame_estimated_poses.push_back(cur_optical_frame_pose_quaternion);

            auto time_stamp_rgb = atof(rgb_vec[0].c_str());
            auto time_stamp_depth = atof(depth_vec[0].c_str());

            auto vo_time_stamp = ((time_stamp_rgb > time_stamp_depth)?time_stamp_rgb:time_stamp_depth);// + time_elapsed;

            time_stamps.push_back(vo_time_stamp);
            
//            cout << rgb_image_file_name << endl;
//            cout << depth_image_file_name << endl;
//
//            cv::imshow("RGB img", h_cur_img);
//            cv::imshow("DEPTH img", h_cur_depth);
//
//            while (true) {
//
//                int c;
//                c = getchar();
//
//                if ((char) c == 27) break;
//
//            }
            
        }
    }
    
    cgmapping::poses_to_file(time_stamps, optical_rgb_frame_estimated_poses, pose_estimation_file);
    
    cout << "end" << endl;
    
}

TEST(visual_odometry_check, visual_odometry_test_with_line_arguments){
    
    using namespace std;
    using namespace boost;
    
    fstream rgb_stream, depth_stream, groundtruth_stream;
    
    //string database_path("/home/spades/kinetic_ws/src/cgmapping/0_non_code_stuff/datasets/rgbd_dataset_freiburg1_xyz/");
    string database_path(global_argv[1]);
    
    string rgb_file_name = database_path + std::string(global_argv[2]);
    string depth_file_name = database_path + std::string(global_argv[3]);
    string groundtruth_file_name = database_path + std::string(global_argv[4]);
    
    rgb_stream.open(rgb_file_name);
    if(!rgb_stream.is_open())
        cout << "Not possible to open rgb name list file" << endl;
    
    depth_stream.open(depth_file_name);
    if(!depth_stream.is_open())
        cout << "Not possible to open depth name list file" << endl;
    
    groundtruth_stream.open(groundtruth_file_name);
    if(!groundtruth_stream.is_open())
        cout << "Not possible to open groundtruth file" << endl;
    
    cgmapping::syncronize_fstreams_from_rgbd_data_set(rgb_stream, depth_stream, groundtruth_stream);
    
    std::string pose_estimation_file(global_argv[5]);
    
    pose_estimation_file = database_path + pose_estimation_file;
    
    string rgb_image_file_name;
    string depth_image_file_name;
    string groundtruth_init_pos;
    
    vector<string> rgb_vec;
    vector<string> depth_vec;
    vector<string> groundtruth_vec;
    
    getline(rgb_stream, rgb_image_file_name);
    getline(depth_stream, depth_image_file_name);
    getline(groundtruth_stream, groundtruth_init_pos);
    
    split_regex(rgb_vec, rgb_image_file_name, regex(" "));
    split_regex(depth_vec, depth_image_file_name, regex(" "));
    split_regex(groundtruth_vec, groundtruth_init_pos, regex(" "));
    
    rgb_image_file_name = database_path + rgb_vec[1];
    depth_image_file_name = database_path + depth_vec[1];
    
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    cv::Mat h_cur_img, h_cur_depth;
    
    h_cur_img = cv::imread(rgb_image_file_name, cv::IMREAD_GRAYSCALE);
    h_cur_depth = cv::imread(depth_image_file_name, CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);
    
    cgmapping::Timer __timer;
    vector<double> time_stamps;
    
    Eigen::Matrix4d cur_optical_frame_pose;
    geometry_msgs::Pose cur_optical_frame_pose_quaternion;
    
    cgmapping::pose_line_2_eigen_pose_matrix4d(groundtruth_init_pos, cur_optical_frame_pose);
    cgmapping::eigen_pose_matrix4d_2_ros_geometry_msg_pose(cur_optical_frame_pose, cur_optical_frame_pose_quaternion);
    
    vector<geometry_msgs::Pose> optical_rgb_frame_estimated_poses;
    
    cudaStream_t raw_strm[4];
    cv::cuda::Stream strm[4];
    
    for (int i = 0; i < 4; ++i) {
        strm[i] = cv::cuda::StreamAccessor::wrapStream(raw_strm[i]);
    }
    
    cgmapping::rgb_d_camera_model camera_model(525.0, 525.0, 319.5, 239.5, 5000);
    
    double eps_error = atof(global_argv[6]);
    uint max_iterations = (uint)atoi(global_argv[7]);
    double t_student_degrees_of_freedom = atof(global_argv[8]);
    double eps_acceptance_std_dev_t_student = atof(global_argv[9]);
    
    cgmapping::image_size_t min_size = static_cast<cgmapping::image_size_t>(atoi(global_argv[10]));
    cgmapping::image_size_t max_size = static_cast<cgmapping::image_size_t>(atoi(global_argv[11]));
    
    assert(min_size < max_size);
    
    cgmapping::cuda::visual_odometer<double> __vo(h_cur_img,
                                                  h_cur_depth,
                                                  camera_model,
                                                  eps_error,
                                                  max_iterations,
                                                  t_student_degrees_of_freedom,
                                                  eps_acceptance_std_dev_t_student,
                                                  cur_optical_frame_pose.data());
    
    time_stamps.push_back(atof(groundtruth_vec[0].c_str()));
    optical_rgb_frame_estimated_poses.push_back(cur_optical_frame_pose_quaternion);

//    cv::namedWindow("RGB img", cv::IMREAD_GRAYSCALE);
//    cv::namedWindow("DEPTH img", CV_16UC1);
    
//    cout << rgb_image_file_name << endl;
//    cout << depth_image_file_name << endl;
    
    int num_estimations = 0;
    float frequency_sum = 0;
    
    cout << "PARAMETERS SET TO " << endl;
    cout << "dataset = " << database_path << endl;
    cout << "rgb file list = " << rgb_file_name << endl;
    cout << "eps_error = " << eps_error << endl;
    cout << "min img size = " << min_size << endl;
    cout << "max img size = " << max_size << endl;
    getchar();
    
    while(getline(rgb_stream, rgb_image_file_name) &&  getline(depth_stream, depth_image_file_name)){
        
        if(!rgb_stream.eof() && !depth_stream.eof() && rgb_image_file_name[0] != '#' && depth_image_file_name[0] != '#') {
            
            rgb_vec.clear();
            split_regex(rgb_vec, rgb_image_file_name, regex(" "));
            rgb_image_file_name = database_path + rgb_vec[1];
            
            depth_vec.clear();
            split_regex(depth_vec, depth_image_file_name, regex(" "));
            depth_image_file_name = database_path + depth_vec[1];
            
            h_cur_img = cv::imread(rgb_image_file_name, cv::IMREAD_GRAYSCALE);
            h_cur_depth = cv::imread(depth_image_file_name, CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);
            
            __timer.reset();
            
            __vo._estimateCamera_motion(h_cur_img, h_cur_depth, min_size, max_size);
            
            auto time_elapsed = __timer.elapsed_in_sec();
            
            frequency_sum += 1./time_elapsed;
            num_estimations++;
            
            __vo._getCur_estimated_pose()._downloadData(cur_optical_frame_pose.data(), 16, &raw_strm[0]);
            
            strm[0].waitForCompletion();
            
            std::cout << cur_optical_frame_pose << std::endl;
            
            std::cout << std::endl;
            std::cout << "avarage frequency = " << frequency_sum/num_estimations << std::endl;
            std::cout << "time elapsed = " << time_elapsed << std::endl;
            std::cout << "instant frequency = " << 1./time_elapsed << std::endl;
            std::cout << "num_its = "  << __vo._getIterations_at_img_resolution(cgmapping::HEX_SIZE) << std::endl;
            std::cout << "num_its = "  << __vo._getIterations_at_img_resolution(cgmapping::OCT_SIZE) << std::endl;
            std::cout << "num_its = "  << __vo._getIterations_at_img_resolution(cgmapping::QUARTER_SIZE) << std::endl;
            std::cout << "num_its = "  << __vo._getIterations_at_img_resolution(cgmapping::HALF_SIZE) << std::endl;
            std::cout << "num_its = "  << __vo._getIterations_at_img_resolution(cgmapping::ORIGINAL_SIZE) << std::endl;
            std::cout << std::endl;
            
            cgmapping::eigen_pose_matrix4d_2_ros_geometry_msg_pose(cur_optical_frame_pose, cur_optical_frame_pose_quaternion);
            
            optical_rgb_frame_estimated_poses.push_back(cur_optical_frame_pose_quaternion);
            
            auto time_stamp_rgb = atof(rgb_vec[0].c_str());
            auto time_stamp_depth = atof(depth_vec[0].c_str());
            
            auto vo_time_stamp = ((time_stamp_rgb > time_stamp_depth)?time_stamp_rgb:time_stamp_depth);// + time_elapsed;
            
            time_stamps.push_back(vo_time_stamp);

//            cout << rgb_image_file_name << endl;
//            cout << depth_image_file_name << endl;
//
//            cv::imshow("RGB img", h_cur_img);
//            cv::imshow("DEPTH img", h_cur_depth);
//
//            while (true) {
//
//                int c;
//                c = getchar();
//
//                if ((char) c == 27) break;
//
//            }
        
        }
    }
    
    cgmapping::poses_to_file(time_stamps, optical_rgb_frame_estimated_poses, pose_estimation_file);
    
    cout << "end" << endl;
    
}

int main(int argc, char **argv) {
    
    ::testing::InitGoogleTest(&argc, argv);

    global_argc = argc;
    global_argv = argv;
    
    auto cublas_stat = cuBLAS_wrapper::cublas_wrapper::_start_cublas_handle_wrapper();
//    cuSOLVER_wrapper::cusolver_wrapper::_start_cusolverDn_handle_wrapper();
    cuBLAS_wrapper::cublas_wrapper::_cublasCheckErrors(cublas_stat, __FILE__, __FUNCTION__, __LINE__);
    
    cuInit(0);
    
    return RUN_ALL_TESTS();

    
}