//
// Created by spades on 05/10/17.
//

#include <cgmapping/cgmapping_utils.h>

double cgmapping::calculate_standard_deviation_t_student(cuLiNA::culina_base_matrix<double> &data,
                                                         double degrees_of_freedom,
                                                         double standard_deviation_initial_guess,
                                                         double acceptance_epsilon,
                                                         cudaStream_t *stream) {
    
    
    double standard_deviation_k_minus_1;
    double standard_deviation_k = standard_deviation_initial_guess;
    
    int number_of_valid_data = (int) cgmapping::cuda::count_valid_data(data, stream);;
    
    do {
    
        standard_deviation_k_minus_1 = standard_deviation_k;
        
        standard_deviation_k = cgmapping::cuda::calculate_standart_deviation_t_student_step(data,
                                                                                            degrees_of_freedom,
                                                                                            standard_deviation_k_minus_1,
                                                                                            number_of_valid_data,
                                                                                            stream);
        
    } while (abs(standard_deviation_k - standard_deviation_k_minus_1) > acceptance_epsilon);
    
    
    return standard_deviation_k;
    
    
}