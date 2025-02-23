cmake_minimum_required(VERSION 3.28 FATAL_ERROR)
#cmake_policy(SET CMP0002 NEW)

#project(TorchExtension LANGUAGES CXX)

#TODO ----vvvv---- TorchExtension MODULE DEFINITION ----vvvv----

project(TorchExtension VERSION 0.1.0 LANGUAGES CXX)

set(LIBRARY_SOURCE_FILES
        src/utils/archiver.cpp
        src/utils/downloader.cpp
        src/utils/md5.cpp
        src/media/image/processing.cpp

        src/models/cnn/alexnet.cpp
        src/models/cnn/lenet5.cpp
        src/models/cnn/vggnet16.cpp
        src/models/cnn/resnet.cpp
        src/models/cnn/unet.cpp
        src/models/cnn/mobilenetv3.cpp

        src/datasets/base.cpp
        src/datasets/caltech.cpp
        src/datasets/celeba.cpp
        src/datasets/cifar.cpp
        src/datasets/cityscapes.cpp
        src/datasets/coco.cpp
        src/datasets/country.cpp
        src/datasets/cre-stereo.cpp
        src/datasets/dtd.cpp
        src/datasets/eth-3d-stereo.cpp
        src/datasets/euro-sat.cpp
        src/datasets/fake-data.cpp
        src/datasets/falling-things-stereo.cpp
        src/datasets/fer.cpp
        src/datasets/fgvc-aircraft.cpp
        src/datasets/flickr.cpp
        src/datasets/flowers.cpp
        src/datasets/flying.cpp
        src/datasets/food.cpp
        src/datasets/gtsrb.cpp
        src/datasets/hd1k.cpp
        src/datasets/hmdb.cpp
        src/datasets/imagenet.cpp
        src/datasets/imagenette.cpp
        src/datasets/in-stereo.cpp
        src/datasets/inaturalist.cpp
        src/datasets/kinetics.cpp
        src/datasets/kitti.cpp
        src/datasets/lfw.cpp
        src/datasets/lsun.cpp
        src/datasets/middlebury.cpp
        src/datasets/mnist.cpp
        src/datasets/moving-mnist.cpp
        src/datasets/omniglot.cpp
        src/datasets/oxfordIII-t-pet.cpp
        src/datasets/pcam.cpp
        src/datasets/photo-tour.cpp
        src/datasets/places.cpp
        src/datasets/rendered-sst2.cpp
        src/datasets/sb-dataset.cpp
        src/datasets/sbu.cpp
        src/datasets/scene-flow-stereo.cpp
        src/datasets/semeion.cpp
        src/datasets/sintel.cpp
        src/datasets/stanford-cars.cpp
        src/datasets/stl.cpp
        src/datasets/sun.cpp
        src/datasets/svhn.cpp
        src/datasets/ucf.cpp
        src/datasets/usps.cpp
        src/datasets/voc.cpp
        src/datasets/wider-face.cpp
)

#add_executable( ${CMAKE_PROJECT_NAME} main.cpp src/datasets/cifar100.cpp src/datasets/cifar10.cpp src/datasets/fashion-mnist.cpp)
add_library(${CMAKE_PROJECT_NAME} SHARED ${LIBRARY_SOURCE_FILES})

#set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_STANDARD_REQUIRED True)

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

#find_package(XZ REQUIRED)



include_directories(/usr/include)
include_directories(${ZLIB_INCLUDE_DIRS})
include_directories(${CURL_INCLUDE_DIR})
include_directories(${ZIP_INCLUDE_DIRS})
#include_directories(${XZ_INCLUDE_DIRS})

target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS})
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
#        XZ::XZ
)

#target_link_libraries(my_executable OpenSSL::SSL OpenSSL::Crypto)


#TODO ----vvvv---- EXPORT MODULE  ----vvvv----

set(DEFAULT_BUILD_TYPE "Release")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

include(GNUInstallDirs)

#set(SOURCE_FILES src/library.cpp)

#add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES})

target_include_directories(${PROJECT_NAME} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

set_target_properties(${PROJECT_NAME} PROPERTIES
        VERSION ${PROJECT_VERSION}
        SOVERSION 1)

install(TARGETS ${PROJECT_NAME} EXPORT TorchExtensionConfig
        ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME})

install(EXPORT TorchExtensionConfig DESTINATION share/TorchExtension/cmake)

export(TARGETS ${PROJECT_NAME} FILE TorchExtensionConfig.cmake)



#message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")
#message(STATUS "TORCH_LIBRARIES: ${TORCH_LIBRARIES}")
#message(STATUS "TORCH_INCLUDE_DIRS: ${TORCH_INCLUDE_DIRS}")


#TODO ----vvvv---- TEST PROJECT ----vvvv----

#add_subdirectory(examples/test)

#cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(Test)
list(APPEND CMAKE_PREFIX_PATH /home/kami/libs/c++/libtorch/)
find_package(Torch REQUIRED)
find_package(TorchExtension REQUIRED)
find_package(LibLZMA REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(TEST_SOURCE_FILES
        src/utils/archiver.cpp
        src/utils/downloader.cpp
        src/media/image/processing.cpp
        src/datasets/mnist.cpp
        src/utils/md5.cpp
        test.cpp
)

add_executable(${CMAKE_PROJECT_NAME} ${TEST_SOURCE_FILES})
target_link_libraries(${CMAKE_PROJECT_NAME} "${TORCH_LIBRARIES}")
#
set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY CXX_STANDARD 17)

target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS})
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

)



project(md)
list(APPEND CMAKE_PREFIX_PATH /home/kami/libs/c++/libtorch/)
find_package(Torch REQUIRED)
find_package(TorchExtension REQUIRED)
find_package(LibLZMA REQUIRED)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)


set(TEST_SOURCE_FILES
        src/utils/md5.cpp
        md.cpp
)

add_executable(${CMAKE_PROJECT_NAME} ${TEST_SOURCE_FILES})
target_link_libraries(${CMAKE_PROJECT_NAME} "${TORCH_LIBRARIES}")
#
set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY CXX_STANDARD 17)

target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS})
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

)



project(md2)
list(APPEND CMAKE_PREFIX_PATH /home/kami/libs/c++/libtorch/)
find_package(Torch REQUIRED)
find_package(TorchExtension REQUIRED)
find_package(LibLZMA REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(TEST_SOURCE_FILES
        src/utils/md5.cpp
        md2.cpp
)

add_executable(${CMAKE_PROJECT_NAME} ${TEST_SOURCE_FILES})
target_link_libraries(${CMAKE_PROJECT_NAME} "${TORCH_LIBRARIES}")
#
set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY CXX_STANDARD 17)

target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS})
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

)



project(xz)
list(APPEND CMAKE_PREFIX_PATH /home/kami/libs/c++/libtorch/)
find_package(Torch REQUIRED)
find_package(TorchExtension REQUIRED)
find_package(LibLZMA REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(TEST_SOURCE_FILES
        xz.cpp
)

add_executable(${CMAKE_PROJECT_NAME} ${TEST_SOURCE_FILES})
target_link_libraries(${CMAKE_PROJECT_NAME} "${TORCH_LIBRARIES}")
#
set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY CXX_STANDARD 17)

target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS})
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
)



project(letest)
list(APPEND CMAKE_PREFIX_PATH /home/kami/libs/c++/libtorch/)
find_package(Torch REQUIRED)
find_package(TorchExtension REQUIRED)
find_package(LibLZMA REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(TEST_SOURCE_FILES
        src/utils/archiver.cpp
        src/utils/downloader.cpp
        src/media/image/processing.cpp
        src/datasets/mnist.cpp
        src/utils/md5.cpp
        letest.cpp
)

add_executable(${CMAKE_PROJECT_NAME} ${TEST_SOURCE_FILES})
target_link_libraries(${CMAKE_PROJECT_NAME} "${TORCH_LIBRARIES}")
#
set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY CXX_STANDARD 17)

target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS})
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
)


