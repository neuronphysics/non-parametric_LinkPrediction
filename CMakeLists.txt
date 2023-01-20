cmake_minimum_required(VERSION 3.23)
project(glfm_cuda_acc CUDA)

set(CMAKE_CUDA_STANDARD 14)
enable_language(C CXX CUDA)

add_executable(glfm_cuda_acc winMain.cpp FakeDrand48.cu ExtendFunction.cu GeneralFunctions.cu InferenceFunctionsNetwork.cu gpuAcc.cu WindowsHeaders.h)

find_package(GSL REQUIRED)
target_link_libraries(glfm_cuda_acc GSL::gsl GSL::gslcblas)


find_package(CUDA)
target_link_libraries(glfm_cuda_acc ${CUDA_CUBLAS_LIBRARIES} ${CUDA_LIBRARIES})

find_package(PThreads4W REQUIRED)
target_link_libraries(glfm_cuda_acc PThreads4W::PThreads4W)

set_target_properties(glfm_cuda_acc PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
