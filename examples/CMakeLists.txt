cmake_minimum_required(VERSION 3.28 FATAL_ERROR)
project(image_classification)
list(APPEND CMAKE_PREFIX_PATH /home/kami/libs/cpp/libtorch/)


#find_package(PkgConfig REQUIRED)
#pkg_check_modules(LIBZIP REQUIRED libzip)
#
#include_directories(${LIBZIP_INCLUDE_DIRS})
#link_directories(${LIBZIP_LIBRARY_DIRS})


find_package(Torch REQUIRED)
find_package(xTorch REQUIRED)
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




