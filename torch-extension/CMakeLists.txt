cmake_minimum_required(VERSION 3.28 FATAL_ERROR)
#cmake_policy(SET CMP0002 NEW)

#project(TorchExtension LANGUAGES CXX)

#TODO ----vvvv---- TorchExtension MODULE DEFINITION ----vvvv----

project(TorchExtension VERSION 0.1.0 LANGUAGES CXX)



# List Of All CPP Files By Category
file(GLOB_RECURSE UTILITY_FILES CONFIGURE_DEPENDS src/utils/*.cpp)
file(GLOB_RECURSE MEDIA_FILES CONFIGURE_DEPENDS src/media/*.cpp)
file(GLOB_RECURSE DATASET_FILES CONFIGURE_DEPENDS src/datasets/*.cpp)
file(GLOB_RECURSE DEFINITION_FILES CONFIGURE_DEPENDS src/definitions/*.cpp)
file(GLOB_RECURSE MODEL_FILES CONFIGURE_DEPENDS src/models/*.cpp)
file(GLOB_RECURSE MODEL_CNN_FILES CONFIGURE_DEPENDS src/models/cnn/*.cpp)

set(LIBRARY_SOURCE_FILES ${UTILITY_FILES} ${MEDIA_FILES} ${DATASET_FILIES} ${DEFINITION_FILES} ${MODEL_FILES} )

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



include_directories(/usr/include)
include_directories(${ZLIB_INCLUDE_DIRS})
include_directories(${CURL_INCLUDE_DIR})
include_directories(${ZIP_INCLUDE_DIRS})


target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS} ${LibArchive_INCLUDE_DIRS})
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
)


#TODO ----vvvv---- EXPORT MODULE  ----vvvv----

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

install(TARGETS ${PROJECT_NAME} EXPORT TorchExtensionConfig
        ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME})

install(EXPORT TorchExtensionConfig DESTINATION share/TorchExtension/cmake)

export(TARGETS ${PROJECT_NAME} FILE TorchExtensionConfig.cmake)

#TODO ----vvvv---- TEST PROJECT ----vvvv----

#add_subdirectory(examples/test)

#cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(Test)
list(APPEND CMAKE_PREFIX_PATH /home/kami/libs/c++/libtorch/)
find_package(Torch REQUIRED)
find_package(TorchExtension REQUIRED)
find_package(LibLZMA REQUIRED)
find_package(LibArchive REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(TEST_SOURCE_FILES ${UTILITY_FILES} ${MEDIA_FILES} src/datasets/mnist.cpp test.cpp)

add_executable(${CMAKE_PROJECT_NAME} ${TEST_SOURCE_FILES})
target_link_libraries(${CMAKE_PROJECT_NAME} "${TORCH_LIBRARIES}")
#
set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY CXX_STANDARD 17)

target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS} ${LibArchive_INCLUDE_DIRS})
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


)



project(md)
list(APPEND CMAKE_PREFIX_PATH /home/kami/libs/c++/libtorch/)
find_package(Torch REQUIRED)
find_package(TorchExtension REQUIRED)
find_package(LibLZMA REQUIRED)
find_package(LibArchive REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)


set(TEST_SOURCE_FILES src/utils/md5.cpp md.cpp)

add_executable(${CMAKE_PROJECT_NAME} ${TEST_SOURCE_FILES})
target_link_libraries(${CMAKE_PROJECT_NAME} "${TORCH_LIBRARIES}")
#
set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY CXX_STANDARD 17)

target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS} ${LibArchive_INCLUDE_DIRS})
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


)



project(md2)
list(APPEND CMAKE_PREFIX_PATH /home/kami/libs/c++/libtorch/)
find_package(Torch REQUIRED)
find_package(TorchExtension REQUIRED)
find_package(LibLZMA REQUIRED)
find_package(LibArchive REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(TEST_SOURCE_FILES src/utils/md5.cpp md2.cpp)

add_executable(${CMAKE_PROJECT_NAME} ${TEST_SOURCE_FILES})
target_link_libraries(${CMAKE_PROJECT_NAME} "${TORCH_LIBRARIES}")
#
set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY CXX_STANDARD 17)

target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS} ${LibArchive_INCLUDE_DIRS})
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


)



project(xz)
list(APPEND CMAKE_PREFIX_PATH /home/kami/libs/c++/libtorch/)
find_package(Torch REQUIRED)
find_package(TorchExtension REQUIRED)
find_package(LibLZMA REQUIRED)
find_package(LibArchive REQUIRED)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(TEST_SOURCE_FILES xz.cpp)

add_executable(${CMAKE_PROJECT_NAME} ${TEST_SOURCE_FILES})
target_link_libraries(${CMAKE_PROJECT_NAME} "${TORCH_LIBRARIES}")
#
set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY CXX_STANDARD 17)

target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS} ${LibArchive_INCLUDE_DIRS})
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

)



project(letest)
list(APPEND CMAKE_PREFIX_PATH /home/kami/libs/c++/libtorch/)
find_package(Torch REQUIRED)
find_package(TorchExtension REQUIRED)
find_package(LibLZMA REQUIRED)
find_package(LibArchive REQUIRED)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(TEST_SOURCE_FILES ${UTILITY_FILES} ${MEDIA_FILES} ${DEFINITION_FILES} src/datasets/mnist.cpp src/models/cnn/lenet5.cpp letest.cpp)

add_executable(${CMAKE_PROJECT_NAME} ${TEST_SOURCE_FILES})
target_link_libraries(${CMAKE_PROJECT_NAME} "${TORCH_LIBRARIES}")
#
set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY CXX_STANDARD 17)

target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS} ${LibArchive_INCLUDE_DIRS})
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

)



project(mnt)
list(APPEND CMAKE_PREFIX_PATH /home/kami/libs/c++/libtorch/)
find_package(Torch REQUIRED)
find_package(TorchExtension REQUIRED)
find_package(LibLZMA REQUIRED)
find_package(LibArchive REQUIRED)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(TEST_SOURCE_FILES ${UTILITY_FILES} ${MEDIA_FILES} ${DEFINITION_FILES} ${DATASET_FILES} src/models/cnn/lenet5.cpp mnt.cpp)

add_executable(${CMAKE_PROJECT_NAME} ${TEST_SOURCE_FILES})
target_link_libraries(${CMAKE_PROJECT_NAME} "${TORCH_LIBRARIES}")
#
set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY CXX_STANDARD 17)

target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS} ${LibArchive_INCLUDE_DIRS})
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

)

####################################### TGZ #######################################

project(tgz)
list(APPEND CMAKE_PREFIX_PATH /home/kami/libs/c++/libtorch/)
find_package(Torch REQUIRED)
find_package(TorchExtension REQUIRED)
find_package(LibLZMA REQUIRED)
find_package(LibArchive REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(TEST_SOURCE_FILES
        src/utils/extract.cpp
        src/utils/downloader.cpp
        src/media/image/processing.cpp
        src/datasets/mnist.cpp
        src/models/cnn/lenet5.cpp
        src/utils/md5.cpp
        include/types/arguments.h
        src/definitions/transforms.cpp
        include/types/enums.h
        tgz.cpp
        include/datasets/base.h
        src/datasets/base.cpp
        include/datasets/functions.h
        src/datasets/functions.cpp
        include/base/datasets.h
        examples/cnn/includes/base.h
        src/media/opencv/images.cpp

)

add_executable(${CMAKE_PROJECT_NAME} ${TEST_SOURCE_FILES})
target_link_libraries(${CMAKE_PROJECT_NAME} "${TORCH_LIBRARIES}")
#
set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY CXX_STANDARD 17)

target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS} ${LibArchive_INCLUDE_DIRS})
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
)


####################################### IMGNT #######################################

project(imgnt)
list(APPEND CMAKE_PREFIX_PATH /home/kami/libs/c++/libtorch/)
find_package(Torch REQUIRED)
find_package(TorchExtension REQUIRED)
find_package(OpenCV REQUIRED)
find_package(LibLZMA REQUIRED)
find_package(LibArchive REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(TEST_SOURCE_FILES
        src/utils/extract.cpp
        src/utils/downloader.cpp
        src/media/image/processing.cpp
        src/datasets/mnist.cpp
        src/media/opencv/images.cpp
        src/media/opencv/videos.cpp
        src/datasets/imagenette.cpp
        src/models/cnn/lenet5.cpp
        src/utils/md5.cpp
        include/types/arguments.h
        src/definitions/transforms.cpp
        include/types/enums.h
        include/datasets/base.h
        src/datasets/base.cpp
        include/datasets/functions.h
        src/datasets/functions.cpp
        include/base/datasets.h
        examples/cnn/includes/base.h
        imgnt.cpp
        include/media/opencv/videos.h

)

add_executable(${CMAKE_PROJECT_NAME} ${TEST_SOURCE_FILES})
target_link_libraries(${CMAKE_PROJECT_NAME} "${TORCH_LIBRARIES}")
#
set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY CXX_STANDARD 17)

include_directories(/usr/include)

target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS} ${LibArchive_INCLUDE_DIRS})
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
)





####################################### VDTEST #######################################

project(vdtest)
list(APPEND CMAKE_PREFIX_PATH /home/kami/libs/c++/libtorch/)
find_package(Torch REQUIRED)
find_package(TorchExtension REQUIRED)
find_package(OpenCV REQUIRED)
find_package(LibLZMA REQUIRED)
find_package(LibArchive REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(TEST_SOURCE_FILES
        src/utils/extract.cpp
        src/utils/downloader.cpp
        src/media/image/processing.cpp
        src/datasets/mnist.cpp
        src/media/opencv/images.cpp
        src/media/opencv/videos.cpp
        src/datasets/imagenette.cpp
        src/models/cnn/lenet5.cpp
        src/utils/md5.cpp
        include/types/arguments.h
        src/definitions/transforms.cpp
        include/types/enums.h
        include/datasets/base.h
        src/datasets/base.cpp
        include/datasets/functions.h
        src/datasets/functions.cpp
        include/base/datasets.h
        examples/cnn/includes/base.h
        vdtest.cpp
        include/media/opencv/videos.h

)

add_executable(${CMAKE_PROJECT_NAME} ${TEST_SOURCE_FILES})
target_link_libraries(${CMAKE_PROJECT_NAME} "${TORCH_LIBRARIES}")
#
set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY CXX_STANDARD 17)

include_directories(/usr/include)

target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS} ${LibArchive_INCLUDE_DIRS})
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
)








####################################### CNTFILES #######################################

project(cntfiles)
list(APPEND CMAKE_PREFIX_PATH /home/kami/libs/c++/libtorch/)
find_package(Torch REQUIRED)
find_package(TorchExtension REQUIRED)
find_package(OpenCV REQUIRED)
find_package(LibLZMA REQUIRED)
find_package(LibArchive REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(TEST_SOURCE_FILES
        src/utils/extract.cpp
        src/utils/downloader.cpp
        src/utils/filesystem.cpp
        src/media/image/processing.cpp
        src/datasets/mnist.cpp
        src/media/opencv/images.cpp
        src/media/opencv/videos.cpp
        src/datasets/imagenette.cpp
        src/models/cnn/lenet5.cpp
        src/utils/md5.cpp
        include/types/arguments.h
        src/definitions/transforms.cpp
        include/types/enums.h
        include/datasets/base.h
        src/datasets/base.cpp
        include/datasets/functions.h
        src/datasets/functions.cpp
        include/base/datasets.h
        examples/cnn/includes/base.h
        include/media/opencv/videos.h
        cntfiles.cpp

)

add_executable(${CMAKE_PROJECT_NAME} ${TEST_SOURCE_FILES})
target_link_libraries(${CMAKE_PROJECT_NAME} "${TORCH_LIBRARIES}")
#
set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY CXX_STANDARD 17)

include_directories(/usr/include)

target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS} ${LibArchive_INCLUDE_DIRS})
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
)











add_subdirectory(examples)

