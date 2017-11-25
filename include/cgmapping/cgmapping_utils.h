//
// Created by spades on 05/10/17.
//

#ifndef CGMAPPING_CGMAPPING_UTILS_H
#define CGMAPPING_CGMAPPING_UTILS_H

#include <cgmapping/cgmapping_utils.cuh>
#include <cuLiNA/culina_base_matrix.h>

namespace cgmapping{
    
    extern double calculate_standard_deviation_t_student(cuLiNA::culina_base_matrix<double> &data,
                                                         double degrees_of_freedom,
                                                         double standard_deviation_initial_guess,
                                                         double acceptance_epsilon,
                                                         cudaStream_t *stream = NULL);

};

#endif //CGMAPPING_CGMAPPING_UTILS_H
