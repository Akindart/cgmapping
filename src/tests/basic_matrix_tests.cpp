//
// Created by spades on 18/12/17.
//

#include <gtest/gtest.h>
#include <cuLiNA/culina_matrix.h>
#include <cuLiNA/culina_operations.h>
#include <cuLiNA/culina_definition.h>

TEST(basic_check, test_eq){
    
    EXPECT_EQ(1, 0);
    
}

TEST(basic_check, test_ne){
    
    EXPECT_NE(1, 0);
    
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
    
    cuLiNA::cuLiNA_error_t stat;
    
    stat = cuLiNA::culina_matrix_Dsum(&A, &B, &C, culiopD);
    
    cudaDeviceSynchronize();
    
    EXPECT_TRUE(C == C_from_matlab);
    
}

int main(int argc, char **argv) {
    
    ::testing::InitGoogleTest(&argc, argv);
    
    cuBLAS_wrapper::cublas_wrapper::_start_cublas_handle_wrapper();
    cuSOLVER_wrapper::cusolver_wrapper::_start_cusolverDn_handle_wrapper();
    
    return RUN_ALL_TESTS();

}