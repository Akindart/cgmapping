cmake_minimum_required(VERSION 2.8.3)
project(cgmapping)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")


## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  cmake_modules
  eigen_conversions
  geometry_msgs
  nav_msgs
  octomap_msgs
  octomap_ros
  pcl_conversions
  pcl_msgs
  pcl_ros
  roscpp
  tf
)

## System dependencies are found with CMake's conventions
find_package(Boost REQUIRED COMPONENTS system)
find_package(Eigen3 REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(OpenCV REQUIRED opencv>=3.2)
find_package(octomap REQUIRED)
find_package(CUDA REQUIRED)

# set CUDA_NVCC_FLAGS as you would do with CXX/C FLAGS
#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -DMY_DEF=1")
#set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMY_DEF=1" )
#set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DMY_DEF=1" )

SET(CMAKE_CXX_FLAGS_DEBUG "-DDEBUG")

SET(CUDA_NVCC_FLAGS "-x cu -arch=sm_30" CACHE STRING "nvcc flags" FORCE)
SET (CUDA_VERBOSE_BUILD ON CACHE BOOL "nvcc verbose" FORCE)
SET(LIB_TYPE STATIC include/cuda_parsing_helper_in_clion/clion_helper.h)
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMY_DEF=1" )
set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DMY_DEF=1" )

## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## To declare and build messages, services or actions from within this
## package, follow these steps:
## * Let MSG_DEP_SET be the set of packages whose message types you use in
##   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
## * In the file package.xml:
##   * add a build_depend tag for "message_generation"
##   * add a build_depend and a run_depend tag for each package in MSG_DEP_SET
##   * If MSG_DEP_SET isn't empty the following dependency has been pulled in
##     but can be declared for certainty nonetheless:
##     * add a run_depend tag for "message_runtime"
## * In this file (CMakeLists.txt):
##   * add "message_generation" and every package in MSG_DEP_SET to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * add "message_runtime" and every package in MSG_DEP_SET to
##     catkin_package(CATKIN_DEPENDS ...)
##   * uncomment the add_*_files sections below as needed
##     and list every .msg/.srv/.action file to be processed
##   * uncomment the generate_messages entry below
##   * add every package in MSG_DEP_SET to generate_messages(DEPENDENCIES ...)

## Generate messages in the 'msg' folder
# add_message_files(
#   FILES
#   Message1.msg
#   Message2.msg
# )

## Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
# generate_messages(
#   DEPENDENCIES
#   geometry_msgs#   nav_msgs#   octomap_msgs#   pcl_msgs
# )

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

## To declare and build dynamic reconfigure parameters within this
## package, follow these steps:
## * In the file package.xml:
##   * add a build_depend and a run_depend tag for "dynamic_reconfigure"
## * In this file (CMakeLists.txt):
##   * add "dynamic_reconfigure" to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * uncomment the "generate_dynamic_reconfigure_options" section below
##     and list every .cfg file to be processed

## Generate dynamic reconfigure parameters in the 'cfg' folder
# generate_dynamic_reconfigure_options(
#   cfg/DynReconf1.cfg
#   cfg/DynReconf2.cfg
# )

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES cgmapping
#  CATKIN_DEPENDS cmake_modules eigen_conversions geometry_msgs nav_msgs octomap_msgs octomap_ros opencv3 pcl_conversions pcl_msgs pcl_ros roscpp tf
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
# include_directories(include)
include_directories(
        include/
        ${catkin_INCLUDE_DIRS}
        ${octomap_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${CUDA_LIBRARIES}
)

## Declare a C++ library
# add_library(cgmapping
#   src/${PROJECT_NAME}/cgmapping.cpp
# )
add_library(cgmapping_lib ${LIB_TYPE}
        include/cgmapping/state6D.h
        include/cgmapping/twist6D.h
        src/cgmapping/twist6D.cpp
        include/cgmapping/se3_lie_algebra_utils.h
        src/cgmapping/se3_lie_algebra_utils.cpp
        include/cgmapping/timer.h
        include/cgmapping/image_pyramid.h
        src/cgmapping/image_pyramid.cpp
        src/cuLiNA/cuSOLVER_wrapper/cusolver_wrapper.cpp
        include/general_utils.h
         src/general_utils.cpp)


cuda_add_library(cuda_aux_lib ${LIB_TYPE}

        include/cuda_parsing_helper_in_clion/clion_helper.h
        include/cuda_device_properties/cuda_device_properties.h
        src/cuda_device_properties/cuda_device_properties.cpp

        )

cuda_add_library(culina_kernels_lib ${LIB_TYPE}

        include/cuLiNA/culina_utils_kernels.cuh
        src/cuLiNA/culina_utils_kernels.cu

        include/cuLiNA/culina_utils.cuh
        src/cuLiNA/culina_utils.cu

        )

add_library(culina_lib ${LIB_TYPE}

        include/cuLiNA/culina_base_matrix.h
        include/cuLiNA/culina_matrix.h
        src/cuLiNA/culina_matrix.cpp

        include/cuLiNA/cuBLAS_wrapper/cublas_wrapper.h
        src/cuLiNA/cuBLAS_wrapper/cublas_wrapper.cpp

        include/cuLiNA/culina_definition.h

        include/cuLiNA/culina_error_data_types.h

        include/cuLiNA/cuSOLVER_wrapper/cusolver_wrapper.h

        include/cuLiNA/culina_operations.h
        src/cuLiNA/culina_operations.cpp
        include/cuLiNA/culina_operation_data_types.h)

cuda_add_library(image_cuda_lib ${LIB_TYPE}

        include/image_pyramid_test_cuda.cuh
        src/tests/opencv_with_cuda_tests/image_pyramid_test_cuda.cu

        )

## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
#add_dependencies(cgmapping_lib ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS} ${LINE_PROFILER_CU_O})
add_dependencies(culina_kernels_lib cuda_aux_lib)
add_dependencies(culina_lib culina_kernels_lib cuda_aux_lib )

# cuda compile GPU_CODE for executable
#  cuda_compile(IMAGE_PYRAMID_TEST src/image_pyramid_test_cuda.cu)

## Declare a C++ executable
cuda_add_executable(cgmapping_node src/cgmapping/cgmapping_node.cpp)

cuda_add_executable(image_pyramid_test_node src/tests/opencv_with_cuda_tests/image_pyramid_test_node.cpp)

cuda_add_executable(twist_coord_test_node src/tests/lie_algebra_twist_coord_tests/twist_coord_test_node.cpp)

cuda_add_executable(cublas_wrapper_test_node src/tests/cublas_wrapper_test/cublas_wrapper_test_node.cpp)

## Add cmake target dependencies of the executable
## same as for the library above
#add_dependencies(cgmapping_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS} ${LINE_PROFILER_CU_O})
#add_dependencies(image_pyramid_test_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS} ${LINE_PROFILER_CU_O})
add_dependencies(image_pyramid_test_node image_cuda_lib cuda_aux_lib cgmapping_lib)
add_dependencies(twist_coord_test_node cgmapping_lib)
add_dependencies(cublas_wrapper_test_node cgmapping_lib culina_lib cuda_aux_lib culina_kernels_lib)

## Specify libraries to link a library or executable target against
target_link_libraries(cgmapping_node
        ${catkin_LIBRARIES}
        ${octomap_LIBRARIES}
)

target_link_libraries(image_pyramid_test_node
        ${catkin_LIBRARIES}
        image_cuda_lib
        cgmapping_lib
        ${OpenCV_LIBRARIES}
        ${CUDA_LIBRARIES}
)

target_link_libraries(twist_coord_test_node
        ${catkin_LIBRARIES}
        ${Eigen3_LIBRARIES}
        cgmapping_lib
)

target_link_libraries(cublas_wrapper_test_node
        ${catkin_LIBRARIES}
        culina_lib
        culina_kernels_lib
        cuda_aux_lib
        cgmapping_lib

        #libraries for parallel linear algebra computing
        -L/usr/local/cuda-8.0/targets/x86_64-linux/lib
        -lcublas
        -lcublas_device
        -lcudart
        -lcusolver
        -fopenmp
        ${CUDA_LIBRARIES}
        )

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# install(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables and/or libraries for installation
# install(TARGETS cgmapping cgmapping_node
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_cgmapping.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
