cmake_minimum_required(VERSION 3.5)
   project(xTorch VERSION 1.0 LANGUAGES CXX)

   # Force C++11 ABI for compatibility with LibTorch
   add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)

   # --- 1. Define Dependency Configuration ---
   set(DEPS_DIR "${CMAKE_SOURCE_DIR}/third_party")
   set(LIBTORCH_DIR "/home/ubuntu/libtorch") # Use locally compiled ARM64 LibTorch
   set(ONNXRUNTIME_DIR "${DEPS_DIR}/onnxruntime")

   # ONNX Runtime for ARM64 CPU (download or build manually)
   set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-aarch64-1.22.0.tgz")
   set(ONNXRUNTIME_SHA256 "bb76395092d150b52c7092dc6b8f2fe4d80f0f3bf0416d2f269193e347e24702") # Replace with actual SHA256

   # Create third_party directory
   if (NOT EXISTS "${DEPS_DIR}")
       file(MAKE_DIRECTORY "${DEPS_DIR}")
   endif ()

   # --- 2. Fetch ONNX Runtime if it doesn't exist locally ---
   if (NOT EXISTS "${ONNXRUNTIME_DIR}")
       message(STATUS "ONNX Runtime not found locally. Downloading and extracting...")
       string(REGEX MATCH "([^/]+)$" ONNXRUNTIME_FILENAME ${ONNXRUNTIME_URL})
       set(ONNXRUNTIME_ARCHIVE "${DEPS_DIR}/${ONNXRUNTIME_FILENAME}")
       file(DOWNLOAD ${ONNXRUNTIME_URL} ${ONNXRUNTIME_ARCHIVE}
               SHOW_PROGRESS)
       message(STATUS "Extracting ONNX Runtime...")
       set(ONNXRUNTIME_TEMP_DIR "${DEPS_DIR}/onnxruntime_temp")
       file(MAKE_DIRECTORY ${ONNXRUNTIME_TEMP_DIR})
       execute_process(
               COMMAND ${CMAKE_COMMAND} -E tar xzf ${ONNXRUNTIME_ARCHIVE}
               WORKING_DIRECTORY ${ONNXRUNTIME_TEMP_DIR}
       )
       file(GLOB EXTRACTED_DIR "${ONNXRUNTIME_TEMP_DIR}/*")
       file(RENAME ${EXTRACTED_DIR} ${ONNXRUNTIME_DIR})
       file(REMOVE_RECURSE ${ONNXRUNTIME_TEMP_DIR})
       file(REMOVE ${ONNXRUNTIME_ARCHIVE})
       message(STATUS "ONNX Runtime setup complete in ${ONNXRUNTIME_DIR}")
   else ()
       message(STATUS "Found local ONNX Runtime at ${ONNXRUNTIME_DIR}")
   endif ()

   # --- 3. Find and Configure Libraries ---
   set(CMAKE_PREFIX_PATH "${LIBTORCH_DIR}")
   find_package(Torch REQUIRED)
   message(STATUS "Configured LibTorch: ${TORCH_VERSION}")

   set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_DIR}/include")
   find_library(
           ONNXRUNTIME_LIBRARY
           NAMES onnxruntime
           PATHS "${ONNXRUNTIME_DIR}/lib"
           REQUIRED
   )
   message(STATUS "Configured ONNX Runtime Library: ${ONNXRUNTIME_LIBRARY}")

   # --- 4. Build Configuration ---
   if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
       set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
       message(STATUS "Setting default build type to: ${CMAKE_BUILD_TYPE}")
   endif ()

   # --- 5. Source File Collection ---
   file(GLOB_RECURSE ACTIVATION_FILES CONFIGURE_DEPENDS src/activations/*.cpp)
   file(GLOB_RECURSE BASE_FILES CONFIGURE_DEPENDS src/base/*.cpp)
   file(GLOB_RECURSE DROPOUT_FILES CONFIGURE_DEPENDS src/dropouts/*.cpp)
   file(GLOB_RECURSE DATA_LOADER_FILES CONFIGURE_DEPENDS src/data_loaders/*.cpp)
   file(GLOB_RECURSIVE DATA_PARALLEL_FILES CONFIGURE_DEPENDS src/data_parallels/*.cpp)
   file(GLOB_RECURSE DATASET_FILES CONFIGURE_DEPENDS src/datasets/*.cpp)
   file(GLOB_RECURSE LOSS_FILES CONFIGURE_DEPENDS src/losses/*.cpp)
   file(GLOB_RECURSE MODEL_FILES CONFIGURE_DEPENDS src/models/*.cpp)
   file(GLOB_RECURSE NORMALIZATION_FILES CONFIGURE_DEPENDS src/normalizations/*.cpp)
   file(GLOB_RECURSE OPTIMIZATION_FILES CONFIGURE_DEPENDS src/optimizations/*.cpp)
   file(GLOB_RECURSE TRAINER_FILES CONFIGURE_DEPENDS src/trainers/*.cpp)
   file(GLOB_RECURSE TRANSFORM_FILES CONFIGURE_DEPENDS src/transforms/*.cpp)
   file(GLOB_RECURSE UTILITY_FILES CONFIGURE_DEPENDS src/utils/*.cpp)
   file(GLOB_RECURSE MATH_FILES CONFIGURE_DEPENDS src/math/*.cpp)
   file(GLOB_RECURSE HEADER_FILES CONFIGURE_DEPENDS include/*.h)

   set(LIBRARY_SOURCE_FILES
           ${ACTIVATION_FILES}
           ${BASE_FILES}
           ${DROPOUT_FILES}
           ${DATA_LOADER_FILES}
           ${DATA_PARALLEL_FILES}
           ${DATASET_FILES}
           ${LOSS_FILES}
           ${MODEL_FILES}
           ${NORMALIZATION_FILES}
           ${OPTIMIZATION_FILES}
           ${TRAINER_FILES}
           ${TRANSFORM_FILES}
           ${UTILITY_FILES}
           ${MATH_FILES}
           path.cpp
   )

   # --- 6. Include Directories ---
   include_directories(SYSTEM ${CMAKE_SOURCE_DIR}/include /usr/include /usr/local/include)

   # --- 7. Library Target Configuration ---
   add_library(xTorch SHARED ${LIBRARY_SOURCE_FILES})
   set_target_properties(xTorch PROPERTIES
           CXX_STANDARD 17
           CXX_STANDARD_REQUIRED YES)

   # --- 8. Dependency Management ---
   find_package(CURL REQUIRED)
   find_package(OpenCV REQUIRED)
   find_package(ZLIB REQUIRED)
   find_package(OpenSSL REQUIRED)
   find_package(LibLZMA REQUIRED)
   find_package(LibArchive REQUIRED)
   find_library(LIBTAR_LIBRARY tar REQUIRED)
   find_path(LIBTAR_INCLUDE_DIR tar.h REQUIRED)
   find_package(glfw3 REQUIRED)
   find_package(OpenGL REQUIRED)
   find_package(Eigen3 REQUIRED)
   find_package(PkgConfig REQUIRED)
   pkg_check_modules(SAMPLERATE REQUIRED samplerate)
   pkg_check_modules(LIBZIP REQUIRED libzip)

   # Build libsndfile
   add_subdirectory(third_party/sndfile)
   set_target_properties(sndfile PROPERTIES POSITION_INDEPENDENT_CODE ON)

   # Build ImGui
   add_library(imgui
           third_party/imgui/imgui.cpp
           third_party/imgui/imgui_draw.cpp
           third_party/imgui/imgui_tables.cpp
           third_party/imgui/imgui_widgets.cpp
           third_party/imgui/imgui_demo.cpp
   )
   target_include_directories(imgui PUBLIC third_party/imgui)

   # ImGui Backend
   add_library(imgui_backend_glfw_gl3
           third_party/imgui/backends/imgui_impl_glfw.cpp
           third_party/imgui/backends/imgui_impl_opengl3.cpp
   )
   target_include_directories(imgui_backend_glfw_gl3 PUBLIC third_party/imgui third_party/imgui/backends)
   target_link_libraries(imgui_backend_glfw_gl3 PUBLIC imgui glfw OpenGL::GL)

   # ImPlot
   add_library(implot
           third_party/implot/implot.cpp
           third_party/implot/implot_items.cpp
   )
   target_include_directories(implot PUBLIC third_party/implot third_party/imgui)
   target_link_libraries(implot PUBLIC imgui)

   # Include directories
   target_include_directories(xTorch
           PUBLIC
           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
           $<INSTALL_INTERFACE:include>
           PRIVATE
           ${TORCH_INCLUDE_DIRS}
           ${ONNXRUNTIME_INCLUDE_DIR}
           ${CURL_INCLUDE_DIR}
           ${OpenCV_INCLUDE_DIRS}
           ${ZLIB_INCLUDE_DIRS}
           ${LibLZMA_INCLUDE_DIRS}
           ${LibArchive_INCLUDE_DIRS}
           ${LIBTAR_INCLUDE_DIR}
           ${LIBZIP_INCLUDE_DIRS}
           ${EIGEN3_INCLUDE_DIR}
           ${SAMPLERATE_INCLUDE_DIRS}
   )

   # Link libraries
   target_link_libraries(xTorch
           PUBLIC
           ${TORCH_LIBRARIES}
           ${OpenCV_LIBS}
           ${LIBZIP_LIBRARIES}
           ${ONNXRUNTIME_LIBRARY}
           PRIVATE
           ${CURL_LIBRARIES}
           ${ZLIB_LIBRARIES}
           ${LIBTAR_LIBRARY}
           OpenSSL::SSL
           OpenSSL::Crypto
           LibLZMA::LibLZMA
           ${LibArchive_LIBRARIES}
           sndfile
           imgui_backend_glfw_gl3
           implot
           ${SAMPLERATE_LIBRARIES}
           Eigen3::Eigen
   )

   # Linker options
   set_target_properties(xTorch PROPERTIES LINK_FLAGS "-Wl,--no-undefined")
   target_link_options(xTorch PRIVATE "-Wl,--no-undefined")

   # --- 9. Installation Configuration ---
   include(GNUInstallDirs)
   install(TARGETS xTorch EXPORT xTorchTargets
           LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
           ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
           RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
           INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
   )
   string(TOLOWER "${PROJECT_NAME}" PROJECT_NAME_LOWER)
   install(FILES xtorch.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME_LOWER})
   install(DIRECTORY include DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME_LOWER})
   install(DIRECTORY third_party DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME_LOWER})
   install(DIRECTORY python/xtorch_py/ DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME_LOWER}/py_modules)
   install(FILES python/requirements.txt DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME_LOWER})

   install(EXPORT xTorchTargets
           FILE xTorchTargets.cmake
           NAMESPACE xTorch::
           DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/xTorch)
   include(CMakePackageConfigHelpers)
   write_basic_package_version_file(
           ${CMAKE_CURRENT_BINARY_DIR}/xTorchConfigVersion.cmake
           VERSION ${PROJECT_VERSION}
           COMPATIBILITY AnyNewerVersion)
   configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/xTorchConfig.cmake.in
           ${CMAKE_CURRENT_BINARY_DIR}/xTorchConfig.cmake
           INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/xTorch)
   install(FILES
           ${CMAKE_CURRENT_BINARY_DIR}/xTorchConfig.cmake
           ${CMAKE_CURRENT_BINARY_DIR}/xTorchConfigVersion.cmake
           DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/xTorch)

   # --- 10. Uninstall Target ---
   if (NOT TARGET uninstall)
       configure_file(
               "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
               "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
               IMMEDIATE @ONLY)
       add_custom_target(uninstall
               COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
   endif ()

   # --- 11. Post-Install Script ---
   install(CODE "
       message(STATUS \"Executing post-install script: setup.sh\")
       execute_process(
           COMMAND sh ${CMAKE_CURRENT_SOURCE_DIR}/scripts/setup.sh \"${CMAKE_INSTALL_PREFIX}\"
           RESULT_VARIABLE res
       )
       if(NOT res EQUAL 0)
           message(FATAL_ERROR \"Post-install script failed with exit code \${res}\")
       endif()
   ")
