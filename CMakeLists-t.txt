cmake_minimum_required(VERSION 3.28 FATAL_ERROR)

project(xTorch VERSION 0.2.0 LANGUAGES CXX)

# List Of All CPP Files By Category
file(GLOB_RECURSE UTILITY_FILES CONFIGURE_DEPENDS src/utils/*.cpp)
file(GLOB_RECURSE MEDIA_FILES CONFIGURE_DEPENDS src/media/*.cpp)
file(GLOB_RECURSE DATASET_FILES CONFIGURE_DEPENDS src/datasets/*.cpp)
file(GLOB_RECURSE DEFINITION_FILES CONFIGURE_DEPENDS src/definitions/*.cpp)
file(GLOB_RECURSE MODEL_FILES CONFIGURE_DEPENDS src/models/*.cpp)
file(GLOB_RECURSE MODEL_CNN_FILES CONFIGURE_DEPENDS src/models/cnn/*.cpp)
file(GLOB_RECURSE DATALOADER_FILES CONFIGURE_DEPENDS src/data-loaders/*.tpp)

set(LIBRARY_SOURCE_FILES ${UTILITY_FILES} ${MEDIA_FILES} ${DATASET_FILIES} ${DEFINITION_FILES} ${MODEL_FILES}  ${DATALOADER_FILES})

add_library(${CMAKE_PROJECT_NAME} SHARED ${LIBRARY_SOURCE_FILES})

find_library(LIBTAR_LIBRARY tar REQUIRED)
find_path(LIBTAR_INCLUDE_DIR tar.h REQUIRED)



if (NOT LIBTAR_LIBRARY OR NOT LIBTAR_INCLUDE_DIR)
    message(FATAL_ERROR "libtar not found. Please install libtar (e.g., sudo apt-get install libtar-dev).")
endif ()

set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY CXX_STANDARD 17)
list(APPEND CMAKE_PREFIX_PATH /home/kami/libs/c++/libtorch/)


find_package(Torch REQUIRED)
find_package(CURL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(ZLIB REQUIRED)

find_package(OpenSSL REQUIRED)
find_package(LibLZMA REQUIRED)
find_package(LibArchive REQUIRED)

find_package(SndFile REQUIRED)

#find_package(PkgConfig REQUIRED)
#pkg_check_modules(LIBSNDFILE REQUIRED libsndfile)

include_directories(/usr/include)
include_directories(${ZLIB_INCLUDE_DIRS})
include_directories(${CURL_INCLUDE_DIR})
include_directories(${ZIP_INCLUDE_DIRS})


target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS} ${LibArchive_INCLUDE_DIRS} ${SndFile_INCLUDE_DIRS} ${LIBSNDFILE_INCLUDE_DIRS})
target_link_libraries(${CMAKE_PROJECT_NAME}
        ${CURL_LIBRARIES}
        ${TORCH_LIBRARIES}
        ${OpenCV_LIBS}
        ${ZLIB_LIBS}
        ${ZIP_LIBRARIES}
        zip
        ${ZLIB_LIBRARIES}
        ${LIBTAR_LIBRARY}
        OpenSSL::SSL
        OpenSSL::Crypto
        LibLZMA::LibLZMA
        ${LibArchive_LIBRARIES}
        ${LIBSNDFILE_LIBRARIES}
)


set(DEFAULT_BUILD_TYPE "Release")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

include(GNUInstallDirs)

target_include_directories(${PROJECT_NAME} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

set_target_properties(${PROJECT_NAME} PROPERTIES
        VERSION ${PROJECT_VERSION}
        SOVERSION 1)

install(TARGETS ${PROJECT_NAME} EXPORT xTorchConfig
        ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME})

install(EXPORT xTorchConfig DESTINATION share/xTorch/cmake)

export(TARGETS ${PROJECT_NAME} FILE xTorchConfig.cmake)




add_subdirectory(examples)
add_subdirectory(temp)

