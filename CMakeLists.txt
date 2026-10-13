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
  foreach(t score bonus turn50 pi_transfer gukjin_state game50 special50
            terminal50 last_card50 settlement50 rollout50 policy50 postchance50
            torch50 hangame_rules50)
    add_executable(cugo_${t}_test tests/${t}_test.cpp)
    target_link_libraries(cugo_${t}_test PRIVATE cugo_core)
    add_test(NAME cugo_${t}_test COMMAND cugo_${t}_test)
  endforeach()

  # Differential replay of games recorded from another implementation (not a test).
  add_executable(cugo_hg_replay50 tools/hg_replay50.cpp)
  target_link_libraries(cugo_hg_replay50 PRIVATE cugo_core)
endif()

if(CUGO_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    if(BUILD_TESTING)
      foreach(t core score bonus deal turn50 pi_transfer gukjin_state game50
                special50 terminal50 last_card50 settlement50 rollout50 policy50
                postchance50 torch50)
        add_executable(cugo_cuda_${t}_test tests/cuda_${t}_test.cu)
        target_link_libraries(cugo_cuda_${t}_test PRIVATE cugo_core)
        set_target_properties(cugo_cuda_${t}_test PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
        if(MSVC)
          target_compile_options(cugo_cuda_${t}_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
        endif()
        add_test(NAME cugo_cuda_${t}_test COMMAND cugo_cuda_${t}_test)
      endforeach()

      foreach(b rollout50 policy50)
        add_executable(cugo_cuda_${b}_bench tests/cuda_${b}_bench.cu)
        target_link_libraries(cugo_cuda_${b}_bench PRIVATE cugo_core)
        set_target_properties(cugo_cuda_${b}_bench PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
        if(MSVC)
          target_compile_options(cugo_cuda_${b}_bench PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
        endif()
      endforeach()
    endif()
  else()
    message(STATUS "CUDA compiler not found; CUDA validation target disabled")
  endif()
endif()
