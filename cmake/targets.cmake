# cmake/targets.cmake
# ===============================================================
#  Defines all library targets for the xTorch project.
#  This file is platform-agnostic.
# ===============================================================

# --- Set Default Build Type ---
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

# --- Collect All Project Source Files ---
file(GLOB_RECURSE ACTIVATION_FILES CONFIGURE_DEPENDS src/activations/*.cpp)
file(GLOB_RECURSE BASE_FILES CONFIGURE_DEPENDS src/base/*.cpp)
file(GLOB_RECURSE DROPOUT_FILES CONFIGURE_DEPENDS src/dropouts/*.cpp)
file(GLOB_RECURSE DATA_LOADER_FILES CONFIGURE_DEPENDS src/data_loaders/*.cpp)
file(GLOB_RECURSE DATA_PARALLEL_FILES CONFIGURE_DEPENDS src/data_parallels/*.cpp)
file(GLOB_RECURSE DATASET_FILES CONFIGURE_DEPENDS src/datasets/*.cpp)
file(GLOB_RECURSE LOSS_FILES CONFIGURE_DEPENDS src/losses/*.cpp)
file(GLOB_RECURSE MODEL_FILES CONFIGURE_DEPENDS src/models/*.cpp)
file(GLOB_RECURSE NORMALIZATION_FILES CONFIGURE_DEPENDS src/normalizations/*.cpp)
file(GLOB_RECURSE OPTIMIZATION_FILES CONFIGURE_DEPENDS src/optimizations/*.cpp)
file(GLOB_RECURSE TRAINER_FILES CONFIGURE_DEPENDS src/trainers/*.cpp)
file(GLOB_RECURSE TRANSFORM_FILES CONFIGURE_DEPENDS src/transforms/*.cpp)
file(GLOB_RECURSE UTILITY_FILES CONFIGURE_DEPENDS src/utils/*.cpp)
file(GLOB_RECURSE MATH_FILES CONFIGURE_DEPENDS src/math/*.cpp)

set(LIBRARY_SOURCE_FILES
        ${ACTIVATION_FILES} ${BASE_FILES} ${DROPOUT_FILES}
        ${DATA_LOADER_FILES} ${DATA_PARALLEL_FILES} ${DATASET_FILES}
        ${LOSS_FILES} ${MODEL_FILES} ${NORMALIZATION_FILES}
        ${OPTIMIZATION_FILES} ${TRAINER_FILES} ${TRANSFORM_FILES}
        ${UTILITY_FILES} ${MATH_FILES} path.cpp
)

# --- Define xTorch Library Target ---
add_library(xTorch SHARED ${LIBRARY_SOURCE_FILES})

target_include_directories(xTorch PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_include_directories(xTorch PRIVATE
        ${TORCH_INCLUDE_DIRS}
        ${ONNXRUNTIME_INCLUDE_DIR}
)

# --- Define Third-Party Library Targets (Built from Source) ---
add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/sndfile)
set_target_properties(sndfile PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(imgui
        third_party/imgui/imgui.cpp
        third_party/imgui/imgui_draw.cpp
        third_party/imgui/imgui_tables.cpp
        third_party/imgui/imgui_widgets.cpp
        third_party/imgui/imgui_demo.cpp
)
target_include_directories(imgui PUBLIC third_party/imgui)


# === THE FIX FOR IMPLOT IS HERE ===
# Apply the same optimization workaround to ImPlot, as it triggers
# the same GCC internal compiler error.
if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_CLANG)
    target_compile_options(implot PRIVATE
            $<$<CONFIG:Release>:-O0>
    )
    message(STATUS "Adding -O0 for ImPlot in Release builds to work around compiler bug.")
endif()
# ==================================



add_library(imgui_backend_glfw_gl3
        third_party/imgui/backends/imgui_impl_glfw.cpp
        third_party/imgui/backends/imgui_impl_opengl3.cpp
)
target_include_directories(imgui_backend_glfw_gl3 PUBLIC third_party/imgui third_party/imgui/backends)
target_link_libraries(imgui_backend_glfw_gl3 PUBLIC imgui glfw OpenGL::GL)

add_library(implot
        third_party/implot/implot.cpp
        third_party/implot/implot_items.cpp
)
target_include_directories(implot PUBLIC third_party/implot third_party/imgui)
target_link_libraries(implot PUBLIC imgui)