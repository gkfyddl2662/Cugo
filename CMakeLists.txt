cmake_minimum_required(VERSION 3.24)
project(Cugo LANGUAGES CXX)

option(CUGO_ENABLE_CUDA "Build CUDA validation targets when a CUDA compiler is available" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(cugo_core INTERFACE)
target_include_directories(cugo_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(cugo_core INTERFACE cxx_std_20)

include(CTest)
if(BUILD_TESTING)
  add_executable(cugo_core_test tests/core_test.cpp)
  target_link_libraries(cugo_core_test PRIVATE cugo_core)
  add_test(NAME cugo_core_test COMMAND cugo_core_test)
endif()

if(CUGO_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    if(BUILD_TESTING)
      add_executable(cugo_cuda_core_test tests/cuda_core_test.cu)
      target_link_libraries(cugo_cuda_core_test PRIVATE cugo_core)
      set_target_properties(cugo_cuda_core_test PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

      add_executable(cugo_cuda_deal_test tests/cuda_deal_test.cu)
      target_link_libraries(cugo_cuda_deal_test PRIVATE cugo_core)
      set_target_properties(cugo_cuda_deal_test PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

      add_executable(cugo_cuda_state_test tests/cuda_state_test.cu)
      target_link_libraries(cugo_cuda_state_test PRIVATE cugo_core)
      set_target_properties(cugo_cuda_state_test PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

      add_executable(cugo_cuda_turn_test tests/cuda_turn_test.cu)
      target_link_libraries(cugo_cuda_turn_test PRIVATE cugo_core)
      set_target_properties(cugo_cuda_turn_test PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

      if(MSVC)
        target_compile_options(cugo_cuda_core_test PRIVATE
          $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
        target_compile_options(cugo_cuda_deal_test PRIVATE
          $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
        target_compile_options(cugo_cuda_state_test PRIVATE
          $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
        target_compile_options(cugo_cuda_turn_test PRIVATE
          $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
      endif()

      add_test(NAME cugo_cuda_core_test COMMAND cugo_cuda_core_test)
      add_test(NAME cugo_cuda_deal_test COMMAND cugo_cuda_deal_test)
      add_test(NAME cugo_cuda_state_test COMMAND cugo_cuda_state_test)
      add_test(NAME cugo_cuda_turn_test COMMAND cugo_cuda_turn_test)
    endif()
  else()
    message(STATUS "CUDA compiler not found; CUDA validation target disabled")
  endif()
endif()
