cmake_minimum_required(VERSION 3.23)
project(glfm_cuda_acc CUDA)

set(CMAKE_CUDA_STANDARD 14)
enable_language(CXX CUDA)

add_executable(glfm_cuda_acc WinMain.cpp ExtendFunction.cu Utils.cu InferenceFunctionsNetwork.cu GpuAcc.cu Log.h GibbsSampler.cu GibbsSampler.cuh Deprecated_Utils.cpp Deprecated_Utils.h UnitTest.cpp UnitTest.h)

find_package(GSL REQUIRED)
target_link_libraries(glfm_cuda_acc GSL::gsl GSL::gslcblas)


find_package(CUDAToolkit)
#target_link_libraries(glfm_cuda_acc ${CUDA_CUBLAS_LIBRARIES} ${CUDA_LIBRARIES} CUDA::cusolver)
target_link_libraries(glfm_cuda_acc ${CUDA_LIBRARIES})

set_target_properties(glfm_cuda_acc PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
