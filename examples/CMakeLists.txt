cmake_minimum_required(VERSION 3.28 FATAL_ERROR)
project(image_classification)
#list(APPEND CMAKE_PREFIX_PATH /home/kami/libs/cpp/libtorch/)




#find_package(PkgConfig REQUIRED)
#pkg_check_modules(LIBZIP REQUIRED libzip)
#
#include_directories(${LIBZIP_INCLUDE_DIRS})
#link_directories(${LIBZIP_LIBRARY_DIRS})


find_package(xTorch REQUIRED)

# Use the correct variable 'xTorch_FOUND' (lowercase 'x')
if(xTorch_FOUND)
    # The xTorch_DIR variable is automatically set by find_package.
    # It points to the directory containing xTorchConfig.cmake,
    # e.g., /usr/local/lib/cmake/xTorch
    message(STATUS "xTorch config directory found at: ${xTorch_DIR}")

    # To find the installation root, we go up a few directories from the config file location.
    # This might require adjustment depending on xTorch's install structure,
    # but /lib/cmake/xTorch is a very common pattern.
    get_filename_component(XTORCH_INSTALL_DIR "${xTorch_DIR}" DIRECTORY) # up to /usr/local/lib/cmake
    get_filename_component(XTORCH_INSTALL_DIR "${XTORCH_INSTALL_DIR}" DIRECTORY) # up to /usr/local/lib
    get_filename_component(XTORCH_INSTALL_DIR "${XTORCH_INSTALL_DIR}" DIRECTORY) # up to /usr/local

    message(STATUS "Deduced xTorch installation root: ${XTORCH_INSTALL_DIR}")

    # Now, construct the path to the bundled libtorch from this reliable root path
    set(BUNDLED_LIBTORCH_PATH "${XTORCH_INSTALL_DIR}/include/xtorch/third_party/libtorch")

    # Check if that directory actually exists before adding it
    if(EXISTS "${BUNDLED_LIBTORCH_PATH}")
        message(STATUS "Found bundled libtorch at: ${BUNDLED_LIBTORCH_PATH}")
        list(PREPEND CMAKE_PREFIX_PATH ${BUNDLED_LIBTORCH_PATH})
    else()
        message(FATAL_ERROR "Could not find the bundled libtorch directory at the expected path: ${BUNDLED_LIBTORCH_PATH}")
    endif()
endif()
find_package(Torch REQUIRED)
find_package(LibLZMA REQUIRED)
find_package(LibArchive REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(CURL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(ZLIB REQUIRED)
find_package(SndFile REQUIRED)
#find_package(LibZip REQUIRED)

#find_package(onnxruntime REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

#set(TEST_SOURCE_FILES ${UTILITY_FILES} ${MEDIA_FILES} ${DEFINITION_FILES} ${DATASET_FILES} ${MODEL_FILES} ${TRANSFORM_FILES})

set(TEST_LIBRARIES ${CURL_LIBRARIES}
        ${CURL_LIBRARIES}
        ${TORCH_LIBRARIES}
        ${OpenCV_LIBS}
        ${ZLIB_LIBS}
        ${ZIP_LIBRARIES}
        zip
        ${ZLIB_LIBRARIES}
        ${LIBTAR_LIBRARY}
        OpenSSL::SSL
        LibLZMA::LibLZMA
        ${LibArchive_LIBRARIES}
        ${XTORCH_LIBRARIES}
        xTorch::xTorch
        OpenSSL::SSL
        OpenSSL::Crypto
        ZLIB::ZLIB
        #        ${ONNXRUNTIME_LIBRARY}
        #        onnxruntime
        ONNX::Runtime
        ${LIBZIP_LIBRARIES}  # Add this line
        #        LibZip::zip

        #        LibZip::zip

)


#add_executable(${PROJECT_NAME} ${TEST_SOURCE_FILES})
#set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

#include_directories(/usr/include usr/local/include/xtorch/)

#target_include_directories(${PROJECT_NAME} PRIVATE ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS} ${LibArchive_INCLUDE_DIRS} ${XTORCH_INCLUDE_DIRS})
#target_link_libraries(${PROJECT_NAME}
#        ${CURL_LIBRARIES}
#        ${TORCH_LIBRARIES}
#        ${OpenCV_LIBS}
#        ${ZLIB_LIBS}
#        ${ZIP_LIBRARIES}
#        zip
#        ${ZLIB_LIBRARIES}
#        ${LIBTAR_LIBRARY}
#        OpenSSL::SSL
#        #        OpenSSL::Crypto
#        LibLZMA::LibLZMA
#        ${LibArchive_LIBRARIES}
#        ${XTORCH_LIBRARIES}
#        xTorch::xTorch
#        ${ONNXRUNTIME_LIBRARY}
#
#)

#========================================= classifying_handwritten_digits_with_lenet_on_mnist =========================================

add_executable(classifying_handwritten_digits_with_lenet_on_mnist ${TEST_SOURCE_FILES} classifying_handwritten_digits_with_lenet_on_mnist.cpp)
target_link_libraries(classifying_handwritten_digits_with_lenet_on_mnist PRIVATE xTorch::xTorch)
set_property(TARGET classifying_handwritten_digits_with_lenet_on_mnist PROPERTY CXX_STANDARD 17)
target_include_directories(classifying_handwritten_digits_with_lenet_on_mnist PRIVATE ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS} ${LibArchive_INCLUDE_DIRS} ${XTORCH_INCLUDE_DIRS})

#========================================= generating_images_with_dcgan =========================================

add_executable(generating_images_with_dcgan ${TEST_SOURCE_FILES} generating_images_with_dcgan.cpp)
target_link_libraries(generating_images_with_dcgan PRIVATE xTorch::xTorch)
set_property(TARGET generating_images_with_dcgan PROPERTY CXX_STANDARD 17)
target_include_directories(generating_images_with_dcgan PRIVATE ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS} ${LibArchive_INCLUDE_DIRS} ${XTORCH_INCLUDE_DIRS})


#========================================= fine_tuning_resnet_on_cifar10 =========================================

#add_executable(fine_tuning_resnet_on_cifar10 ${TEST_SOURCE_FILES} fine_tuning_resnet_on_cifar10.cpp)
#target_link_libraries(fine_tuning_resnet_on_cifar10 "${TORCH_LIBRARIES}")
#set_property(TARGET fine_tuning_resnet_on_cifar10 PROPERTY CXX_STANDARD 17)
#target_include_directories(fine_tuning_resnet_on_cifar10 PRIVATE ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS} ${LibArchive_INCLUDE_DIRS})
#target_link_libraries(fine_tuning_resnet_on_cifar10 ${TEST_LIBRARIES})
#target_include_directories(fine_tuning_resnet_on_cifar10 PRIVATE ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS} ${LibArchive_INCLUDE_DIRS} ${XTORCH_INCLUDE_DIRS})
#target_link_libraries(fine_tuning_resnet_on_cifar10 ${TEST_LIBRARIES})




