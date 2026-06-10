cmake_minimum_required(VERSION 3.21) # Bumped to 3.21 for $<TARGET_RUNTIME_DLLS>
project(Connect4_CPP LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- LibTorch Configuration ---
set(LIBTORCH_DIR "D:/libtorch")
set(CMAKE_PREFIX_PATH "${LIBTORCH_DIR}")
set(CAFFE2_USE_CUDNN ON CACHE BOOL "Force cuDNN for Caffe2" FORCE)

find_package(Torch REQUIRED)

# --- Dependencies ---
include(FetchContent)

FetchContent_Declare(
    tsl_robin_map
    GIT_REPOSITORY https://github.com/Tessil/robin-map.git
    GIT_TAG        v1.4.1
)
FetchContent_MakeAvailable(tsl_robin_map)

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.15.0
)
FetchContent_MakeAvailable(Catch2)

# --- Source Files ---
# Note: In strict production environments, it's better to explicitly list files 
# rather than using GLOB, so CMake knows to reconfigure when new files are added.
file(GLOB_RECURSE GAME_LOGIC_SOURCES 
    "src/*.cpp" 
    "src/*.h"
)

# Exclude entry points from the shared logic list
list(REMOVE_ITEM GAME_LOGIC_SOURCES 
    "${CMAKE_CURRENT_SOURCE_DIR}/src/training_loop.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/test_connect4.cpp"
)

# -------------------------------------------------------------
# Helper Function: Copy DLLs to Target Directory
# -------------------------------------------------------------
function(copy_target_dlls target_name)
    if(WIN32)
        # 1. Hardcoded DLLs (cuDNN, OpenMP, CUPTI, etc.)
        set(EXTRA_DLLS
            "${LIBTORCH_DIR}/lib/cudnn64_9.dll"
            "${LIBTORCH_DIR}/lib/cudnn_adv64_9.dll"
            "${LIBTORCH_DIR}/lib/cudnn_cnn64_9.dll"
            "${LIBTORCH_DIR}/lib/cudnn_ops64_9.dll"
            "${LIBTORCH_DIR}/lib/cudnn_graph64_9.dll"
            "${LIBTORCH_DIR}/lib/libiomp5md.dll"
            "${LIBTORCH_DIR}/lib/cupti64_2025.3.0.dll"
            "${LIBTORCH_DIR}/lib/cudnn_engines_runtime_compiled64_9.dll"
            "${LIBTORCH_DIR}/lib/cudnn_engines_precompiled64_9.dll"
            "${LIBTORCH_DIR}/lib/cudnn_heuristic64_9.dll"
        )
        
        # Filter out missing files to prevent build errors if some DLLs aren't present
        set(VALID_EXTRA_DLLS "")
        foreach(dll ${EXTRA_DLLS})
            if(EXISTS "${dll}")
                list(APPEND VALID_EXTRA_DLLS "${dll}")
            endif()
        endforeach()

        if(VALID_EXTRA_DLLS)
            add_custom_command(
                TARGET ${target_name} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${VALID_EXTRA_DLLS}
                    $<TARGET_FILE_DIR:${target_name}>
                COMMENT "Copying extra LibTorch/cuDNN DLLs for ${target_name}..."
            )
        endif()

        # 2. Automatically detected runtime DLLs
        add_custom_command(
            TARGET ${target_name} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_RUNTIME_DLLS:${target_name}>
                $<TARGET_FILE_DIR:${target_name}>
            COMMAND_EXPAND_LISTS
            COMMENT "Copying runtime DLLs for ${target_name}..."
        )
    endif()
endfunction()

# -------------------------------------------------------------
# 1. Target: Main Application
# -------------------------------------------------------------
add_executable(Connect4AlphaZero 
    "${CMAKE_CURRENT_SOURCE_DIR}/src/training_loop.cpp" 
    ${GAME_LOGIC_SOURCES}
)

target_link_libraries(Connect4AlphaZero PRIVATE 
    ${TORCH_LIBRARIES} 
    tsl::robin_map
)

if(MSVC)
    target_compile_options(Connect4AlphaZero PRIVATE ${TORCH_CXX_FLAGS})
endif()

# Separate output directory for the main app
set_target_properties(Connect4AlphaZero PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/main"
)

# Apply the DLL copy logic to the main app
copy_target_dlls(Connect4AlphaZero)

# -------------------------------------------------------------
# 2. Target: Tests
# -------------------------------------------------------------
enable_testing()
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)

add_executable(Connect4Tests 
    "${CMAKE_CURRENT_SOURCE_DIR}/src/test_connect4.cpp" 
    ${GAME_LOGIC_SOURCES}
)

target_link_libraries(Connect4Tests PRIVATE 
    ${TORCH_LIBRARIES} 
    tsl::robin_map 
    Catch2::Catch2WithMain
)

if(MSVC)
    target_compile_options(Connect4Tests PRIVATE ${TORCH_CXX_FLAGS})
endif()

# Separate output directory for tests
set_target_properties(Connect4Tests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/tests"
)

# IMPORTANT: Copy DLLs BEFORE catch_discover_tests. Catch2 runs the executable 
# at build time to discover tests, and it needs the DLLs to be present to run!
copy_target_dlls(Connect4Tests)

# Register tests with CTest
catch_discover_tests(Connect4Tests
    TEST_SPECIFIC_ENV "PATH=${LIBTORCH_DIR}/lib;$ENV{PATH}"
)