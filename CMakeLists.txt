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

  add_executable(cugo_score_test tests/score_test.cpp)
  target_link_libraries(cugo_score_test PRIVATE cugo_core)
  add_test(NAME cugo_score_test COMMAND cugo_score_test)

  add_executable(cugo_bonus_test tests/bonus_test.cpp)
  target_link_libraries(cugo_bonus_test PRIVATE cugo_core)
  add_test(NAME cugo_bonus_test COMMAND cugo_bonus_test)

  add_executable(cugo_turn50_test tests/turn50_test.cpp)
  target_link_libraries(cugo_turn50_test PRIVATE cugo_core)
  add_test(NAME cugo_turn50_test COMMAND cugo_turn50_test)

  add_executable(cugo_pi_transfer_test tests/pi_transfer_test.cpp)
  target_link_libraries(cugo_pi_transfer_test PRIVATE cugo_core)
  add_test(NAME cugo_pi_transfer_test COMMAND cugo_pi_transfer_test)

  add_executable(cugo_gukjin_state_test tests/gukjin_state_test.cpp)
  target_link_libraries(cugo_gukjin_state_test PRIVATE cugo_core)
  add_test(NAME cugo_gukjin_state_test COMMAND cugo_gukjin_state_test)

  add_executable(cugo_game50_test tests/game50_test.cpp)
  target_link_libraries(cugo_game50_test PRIVATE cugo_core)
  add_test(NAME cugo_game50_test COMMAND cugo_game50_test)

  add_executable(cugo_special50_test tests/special50_test.cpp)
  target_link_libraries(cugo_special50_test PRIVATE cugo_core)
  add_test(NAME cugo_special50_test COMMAND cugo_special50_test)

  add_executable(cugo_terminal50_test tests/terminal50_test.cpp)
  target_link_libraries(cugo_terminal50_test PRIVATE cugo_core)
  add_test(NAME cugo_terminal50_test COMMAND cugo_terminal50_test)

  add_executable(cugo_last_card50_test tests/last_card50_test.cpp)
  target_link_libraries(cugo_last_card50_test PRIVATE cugo_core)
  add_test(NAME cugo_last_card50_test COMMAND cugo_last_card50_test)

  add_executable(cugo_settlement50_test tests/settlement50_test.cpp)
  target_link_libraries(cugo_settlement50_test PRIVATE cugo_core)
  add_test(NAME cugo_settlement50_test COMMAND cugo_settlement50_test)

  add_executable(cugo_rollout50_test tests/rollout50_test.cpp)
  target_link_libraries(cugo_rollout50_test PRIVATE cugo_core)
  add_test(NAME cugo_rollout50_test COMMAND cugo_rollout50_test)

  add_executable(cugo_policy50_test tests/policy50_test.cpp)
  target_link_libraries(cugo_policy50_test PRIVATE cugo_core)
  add_test(NAME cugo_policy50_test COMMAND cugo_policy50_test)

  add_executable(cugo_postchance50_test tests/postchance50_test.cpp)
  target_link_libraries(cugo_postchance50_test PRIVATE cugo_core)
  add_test(NAME cugo_postchance50_test COMMAND cugo_postchance50_test)
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

      add_executable(cugo_cuda_score_test tests/cuda_score_test.cu)
      target_link_libraries(cugo_cuda_score_test PRIVATE cugo_core)
      set_target_properties(cugo_cuda_score_test PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

      add_executable(cugo_cuda_bonus_test tests/cuda_bonus_test.cu)
      target_link_libraries(cugo_cuda_bonus_test PRIVATE cugo_core)
      set_target_properties(cugo_cuda_bonus_test PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

      add_executable(cugo_cuda_deal_test tests/cuda_deal_test.cu)
      target_link_libraries(cugo_cuda_deal_test PRIVATE cugo_core)
      set_target_properties(cugo_cuda_deal_test PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

      add_executable(cugo_cuda_state_test tests/cuda_state_test.cu)
      target_link_libraries(cugo_cuda_state_test PRIVATE cugo_core)
      set_target_properties(cugo_cuda_state_test PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

      add_executable(cugo_cuda_turn_test tests/cuda_turn_test.cu)
      target_link_libraries(cugo_cuda_turn_test PRIVATE cugo_core)
      set_target_properties(cugo_cuda_turn_test PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

      add_executable(cugo_cuda_turn50_test tests/cuda_turn50_test.cu)
      target_link_libraries(cugo_cuda_turn50_test PRIVATE cugo_core)
      set_target_properties(cugo_cuda_turn50_test PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

      add_executable(cugo_cuda_pi_transfer_test tests/cuda_pi_transfer_test.cu)
      target_link_libraries(cugo_cuda_pi_transfer_test PRIVATE cugo_core)
      set_target_properties(cugo_cuda_pi_transfer_test PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

      add_executable(cugo_cuda_gukjin_state_test tests/cuda_gukjin_state_test.cu)
      target_link_libraries(cugo_cuda_gukjin_state_test PRIVATE cugo_core)
      set_target_properties(cugo_cuda_gukjin_state_test PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

      add_executable(cugo_cuda_game50_test tests/cuda_game50_test.cu)
      target_link_libraries(cugo_cuda_game50_test PRIVATE cugo_core)
      set_target_properties(cugo_cuda_game50_test PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

      add_executable(cugo_cuda_special50_test tests/cuda_special50_test.cu)
      target_link_libraries(cugo_cuda_special50_test PRIVATE cugo_core)
      set_target_properties(cugo_cuda_special50_test PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

      add_executable(cugo_cuda_terminal50_test tests/cuda_terminal50_test.cu)
      target_link_libraries(cugo_cuda_terminal50_test PRIVATE cugo_core)
      set_target_properties(cugo_cuda_terminal50_test PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

      add_executable(cugo_cuda_last_card50_test tests/cuda_last_card50_test.cu)
      target_link_libraries(cugo_cuda_last_card50_test PRIVATE cugo_core)
      set_target_properties(cugo_cuda_last_card50_test PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

      add_executable(cugo_cuda_settlement50_test tests/cuda_settlement50_test.cu)
      target_link_libraries(cugo_cuda_settlement50_test PRIVATE cugo_core)
      set_target_properties(cugo_cuda_settlement50_test PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

      add_executable(cugo_cuda_rollout50_test tests/cuda_rollout50_test.cu)
      target_link_libraries(cugo_cuda_rollout50_test PRIVATE cugo_core)
      set_target_properties(cugo_cuda_rollout50_test PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

      add_executable(cugo_cuda_rollout50_bench tests/cuda_rollout50_bench.cu)
      target_link_libraries(cugo_cuda_rollout50_bench PRIVATE cugo_core)
      set_target_properties(cugo_cuda_rollout50_bench PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

      add_executable(cugo_cuda_policy50_test tests/cuda_policy50_test.cu)
      target_link_libraries(cugo_cuda_policy50_test PRIVATE cugo_core)
      set_target_properties(cugo_cuda_policy50_test PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

      add_executable(cugo_cuda_policy50_bench tests/cuda_policy50_bench.cu)
      target_link_libraries(cugo_cuda_policy50_bench PRIVATE cugo_core)
      set_target_properties(cugo_cuda_policy50_bench PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

      add_executable(cugo_cuda_postchance50_test tests/cuda_postchance50_test.cu)
      target_link_libraries(cugo_cuda_postchance50_test PRIVATE cugo_core)
      set_target_properties(cugo_cuda_postchance50_test PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

      if(MSVC)
        target_compile_options(cugo_cuda_core_test PRIVATE
          $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
        target_compile_options(cugo_cuda_score_test PRIVATE
          $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
        target_compile_options(cugo_cuda_bonus_test PRIVATE
          $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
        target_compile_options(cugo_cuda_deal_test PRIVATE
          $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
        target_compile_options(cugo_cuda_state_test PRIVATE
          $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
        target_compile_options(cugo_cuda_turn_test PRIVATE
          $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
        target_compile_options(cugo_cuda_turn50_test PRIVATE
          $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
        target_compile_options(cugo_cuda_pi_transfer_test PRIVATE
          $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
        target_compile_options(cugo_cuda_gukjin_state_test PRIVATE
          $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
        target_compile_options(cugo_cuda_game50_test PRIVATE
          $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
        target_compile_options(cugo_cuda_special50_test PRIVATE
          $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
        target_compile_options(cugo_cuda_terminal50_test PRIVATE
          $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
        target_compile_options(cugo_cuda_last_card50_test PRIVATE
          $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
        target_compile_options(cugo_cuda_settlement50_test PRIVATE
          $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
        target_compile_options(cugo_cuda_rollout50_test PRIVATE
          $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
        target_compile_options(cugo_cuda_rollout50_bench PRIVATE
          $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
        target_compile_options(cugo_cuda_policy50_test PRIVATE
          $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
        target_compile_options(cugo_cuda_policy50_bench PRIVATE
          $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
        target_compile_options(cugo_cuda_postchance50_test PRIVATE
          $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
      endif()

      add_test(NAME cugo_cuda_core_test COMMAND cugo_cuda_core_test)
      add_test(NAME cugo_cuda_score_test COMMAND cugo_cuda_score_test)
      add_test(NAME cugo_cuda_bonus_test COMMAND cugo_cuda_bonus_test)
      add_test(NAME cugo_cuda_deal_test COMMAND cugo_cuda_deal_test)
      add_test(NAME cugo_cuda_state_test COMMAND cugo_cuda_state_test)
      add_test(NAME cugo_cuda_turn_test COMMAND cugo_cuda_turn_test)
      add_test(NAME cugo_cuda_turn50_test COMMAND cugo_cuda_turn50_test)
      add_test(NAME cugo_cuda_pi_transfer_test COMMAND cugo_cuda_pi_transfer_test)
      add_test(NAME cugo_cuda_gukjin_state_test COMMAND cugo_cuda_gukjin_state_test)
      add_test(NAME cugo_cuda_game50_test COMMAND cugo_cuda_game50_test)
      add_test(NAME cugo_cuda_special50_test COMMAND cugo_cuda_special50_test)
      add_test(NAME cugo_cuda_terminal50_test COMMAND cugo_cuda_terminal50_test)
      add_test(NAME cugo_cuda_last_card50_test COMMAND cugo_cuda_last_card50_test)
      add_test(NAME cugo_cuda_settlement50_test COMMAND cugo_cuda_settlement50_test)
      add_test(NAME cugo_cuda_rollout50_test COMMAND cugo_cuda_rollout50_test)
      add_test(NAME cugo_cuda_policy50_test COMMAND cugo_cuda_policy50_test)
      add_test(NAME cugo_cuda_postchance50_test COMMAND cugo_cuda_postchance50_test)
    endif()
  else()
    message(STATUS "CUDA compiler not found; CUDA validation target disabled")
  endif()
endif()
