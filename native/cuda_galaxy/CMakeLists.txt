cmake_minimum_required(VERSION 3.18)
project(dsp_cuda_galaxy LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

add_library(dsp_cuda_galaxy SHARED
    src/dsp_cuda_galaxy.cu
)

target_include_directories(dsp_cuda_galaxy
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_options(dsp_cuda_galaxy
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false --ftz=false --prec-div=true --prec-sqrt=true>
)

set_target_properties(dsp_cuda_galaxy PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)
