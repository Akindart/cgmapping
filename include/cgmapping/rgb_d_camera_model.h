//
// Created by spades on 21/03/18.
//

#ifndef CGMAPPING_RGB_D_CAMERA_MODEL_H
#define CGMAPPING_RGB_D_CAMERA_MODEL_H

namespace cgmapping {
    
    //simple pinhole camera model
    class rgb_d_camera_model {
 
        double focus_x;
        double focus_y;
        double centroid_x;
        double centroid_y;
        
        //this is for when the info in depth image is somehow escalated
        unsigned int scale_for_depth;
        
     public:
        
        rgb_d_camera_model(){};
    
        rgb_d_camera_model(double focus_x,
                           double focus_y,
                           double centroid_x,
                           double centroid_y,
                           unsigned int scale_for_depth)
            : focus_x(focus_x),
              focus_y(focus_y),
              centroid_x(centroid_x),
              centroid_y(centroid_y),
              scale_for_depth(scale_for_depth){};
    
        double _getFocus_x() const {
            return focus_x;
        }
        void _setFocus_x(double focus_x) {
            rgb_d_camera_model::focus_x = focus_x;
        }
        double _getFocus_y() const {
            return focus_y;
        }
        void _setFocus_y(double focus_y) {
            rgb_d_camera_model::focus_y = focus_y;
        }
        double _getCentroid_x() const {
            return centroid_x;
        }
        void _setCentroid_x(double centroid_x) {
            rgb_d_camera_model::centroid_x = centroid_x;
        }
        double _getCentroid_y() const {
            return centroid_y;
        }
        void _setCentroid_y(double centroid_y) {
            rgb_d_camera_model::centroid_y = centroid_y;
        }
        unsigned int _getScale_for_depth() const {
            return scale_for_depth;
        }
        void _setScale_for_depth(unsigned int scale_for_depth) {
            rgb_d_camera_model::scale_for_depth = scale_for_depth;
        }
    
        virtual ~rgb_d_camera_model() {
        
        }
    
    };
    
}

#endif //CGMAPPING_RGB_D_CAMERA_MODEL_H
