cmake_minimum_required(VERSION 3.28 FATAL_ERROR)

project(xTorch VERSION 0.2.0 LANGUAGES CXX)

# ----------------------------------------------------------------------------
# Collect source files
file(GLOB_RECURSE UTILITY_FILES CONFIGURE_DEPENDS src/utils/*.cpp)
file(GLOB_RECURSE MEDIA_FILES CONFIGURE_DEPENDS src/media/*.cpp)
file(GLOB_RECURSE DATASET_FILES CONFIGURE_DEPENDS src/datasets/*.cpp)
file(GLOB_RECURSE DEFINITION_FILES CONFIGURE_DEPENDS src/definitions/*.cpp)
file(GLOB_RECURSE MODEL_FILES CONFIGURE_DEPENDS src/models/*.cpp)
file(GLOB_RECURSE MODEL_CNN_FILES CONFIGURE_DEPENDS src/models/cnn/*.cpp)
file(GLOB_RECURSE DATALOADER_FILES CONFIGURE_DEPENDS src/data-loaders/*.tpp)
file(GLOB_RECURSE TRANSFORM_FILES CONFIGURE_DEPENDS src/transforms/*.cpp)
file(GLOB_RECURSE TEMP_FILES CONFIGURE_DEPENDS src/temp/*.cpp)

set(LIBRARY_SOURCE_FILES
        ${UTILITY_FILES} ${MEDIA_FILES} ${DATASET_FILES} ${DEFINITION_FILES}
        ${MODEL_FILES} ${MODEL_CNN_FILES} ${DATALOADER_FILES}
        ${TRANSFORM_FILES} ${TEMP_FILES})

# ----------------------------------------------------------------------------
# Library Target
add_library(xTorch SHARED ${LIBRARY_SOURCE_FILES})

# Set C++ standard
set_target_properties(xTorch PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# ----------------------------------------------------------------------------
# Dependencies
list(APPEND CMAKE_PREFIX_PATH /home/kami/libs/c++/libtorch/)

find_package(Torch REQUIRED)
find_package(CURL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(ZLIB REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(LibLZMA REQUIRED)
find_package(LibArchive REQUIRED)
find_package(SndFile REQUIRED)

find_library(LIBTAR_LIBRARY tar REQUIRED)
find_path(LIBTAR_INCLUDE_DIR tar.h REQUIRED)

# Include directories
target_include_directories(xTorch
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS}
        ${ZLIB_INCLUDE_DIRS} ${CURL_INCLUDE_DIR}
        ${LibArchive_INCLUDE_DIRS} ${SndFile_INCLUDE_DIRS})

# Link libraries
target_link_libraries(xTorch
        ${TORCH_LIBRARIES} ${CURL_LIBRARIES} ${OpenCV_LIBS} ${ZLIB_LIBRARIES}
        ${LIBTAR_LIBRARY} OpenSSL::SSL OpenSSL::Crypto LibLZMA::LibLZMA
        ${LibArchive_LIBRARIES} ${SndFile_LIBRARIES})

# ----------------------------------------------------------------------------
# Install Setup
include(GNUInstallDirs)

install(TARGETS xTorch EXPORT xTorchTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

string(TOLOWER "${PROJECT_NAME}" PROJECT_NAME_LOWER)


#install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
#install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME})
install(FILES  xtorch.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME_LOWER})
install(DIRECTORY include DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME_LOWER})
install(DIRECTORY third_party DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME_LOWER})


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

# ----------------------------------------------------------------------------
# Uninstall Target
if(NOT TARGET uninstall)
    configure_file(
            "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
            "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
            IMMEDIATE @ONLY)

    add_custom_target(uninstall
            COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
endif()



# -----------------------------------------------------------------------------
# Subdirectories
# -----------------------------------------------------------------------------
# Add subdirectories for example and temporary code
add_subdirectory(examples)
#add_subdirectory(temp)
