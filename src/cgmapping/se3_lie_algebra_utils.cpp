//
// Created by spades on 01/05/18.
//

#include <cgmapping/se3_lie_algebra_utils.cuh>

void cgmapping::logarithmic_Dmap_se3(Matrix4d &homogenic_transformation,
                                     Vector3d &linear_vel,
                                     Vector3d &angular_vel,
                                     double time_elapsed) {

    double theta = acos((homogenic_transformation.block(0,0,3,3).trace()-1.)/2.);
    
    Matrix3d ln_R, V_inverse;
    
    ln_R << (theta/(2*theta))*(homogenic_transformation.block(0,0,3,3) - homogenic_transformation.block(0,0,3,3).transpose());
    
    angular_vel << ln_R(2,1), ln_R(0,2), ln_R(1,0);
    angular_vel = angular_vel/time_elapsed;
    
    double alpha = sin(theta)/theta;
    double one_over_theta_sqrd = 1/(theta*theta);
    double beta = (1 - cos(theta))*one_over_theta_sqrd;
    
    V_inverse.setIdentity();
    
    double A = 0.5;
    double B = alpha;
    
    
    if(theta > NEAR_ZERO_VALUE){
        B = one_over_theta_sqrd*(1 - (alpha/(2*beta)));
    }else{
        
        double theta_sqrd = theta*theta;
        B = (1./12.)*(1 + ((theta_sqrd/60)*(1 + (theta_sqrd/42)*(1 + (theta_sqrd/40)))));
        
    }
    
    V_inverse = V_inverse - A*(ln_R) + B*ln_R*ln_R;
    
    linear_vel = (V_inverse*homogenic_transformation.block(0,3,3,1))/time_elapsed;
    

}
void cgmapping::adjoint_Dse3(Matrix4d &homogenic_transformation, Matrix<double, 6, 6> &adjoint_matrix) {
    
    adjoint_matrix.block(0,3,3,3).setZero();
    
    adjoint_matrix.block(0,0,3,3) << homogenic_transformation.block(0,0,3,3);
    adjoint_matrix.block(3,3,3,3) << homogenic_transformation.block(0,0,3,3);
    
    adjoint_matrix.block(0,3,3,3) << 0, -homogenic_transformation(2,3), homogenic_transformation(1,3),
                                     homogenic_transformation(2,3), 0, -homogenic_transformation(0,3),
                                     -homogenic_transformation(1,3), homogenic_transformation(0,3), 0;
    
    adjoint_matrix.block(0,3,3,3) = adjoint_matrix.block(0,3,3,3)*homogenic_transformation.block(0,0,3,3);

}
