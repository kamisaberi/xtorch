project(examples)
list(APPEND CMAKE_PREFIX_PATH /home/kami/libs/c++/libtorch/)
find_package(Torch REQUIRED)
#find_package(TorchExtension REQUIRED)
find_package(LibLZMA REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(BASE_SOURCE_FILES
        ../src/utils/extract.cpp
        ../src/utils/downloader.cpp
        ../src/media/image/processing.cpp
        ../src/datasets/mnist.cpp
        ../src/models/cnn/lenet5.cpp
        ../src/utils/md5.cpp
        ../include/types/arguments.h
        ../src/definitions/transforms.cpp
        ../include/types/enums.h
        ../include/datasets/base.h
        ../src/datasets/base.cpp
        ../include/datasets/functions.h
        ../src/datasets/functions.cpp
        ../include/base/datasets.h
        ../examples/cnn/includes/base.h
)

set(TEST_LIBRARIES ${CURL_LIBRARIES}
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




# LeNet5 MNIST
set(SOURCE_FILES
        ../src/datasets/mnist.cpp
        ../src/models/cnn/lenet5.cpp
)
add_executable(lenet5-mnist ${BASE_SOURCE_FILES} ${SOURCE_FILES} cnn/lenet5-mnist.cpp)
target_link_libraries(lenet5-mnist "${TORCH_LIBRARIES}")
set_property(TARGET lenet5-mnist PROPERTY CXX_STANDARD 17)
target_include_directories(lenet5-mnist PRIVATE ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS})
target_link_libraries(lenet5-mnist ${TEST_LIBRARIES})

# LeNet5 FashionMNIST
set(SOURCE_FILES
        ../src/datasets/mnist.cpp
        ../src/models/cnn/lenet5.cpp
)
add_executable(lenet5-fashionmnist ${BASE_SOURCE_FILES} ${SOURCE_FILES} cnn/lenet5-fashionmnist.cpp)
target_link_libraries(lenet5-fashionmnist "${TORCH_LIBRARIES}")
set_property(TARGET lenet5-fashionmnist PROPERTY CXX_STANDARD 17)
target_include_directories(lenet5-fashionmnist PRIVATE ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS})
target_link_libraries(lenet5-fashionmnist ${TEST_LIBRARIES})


# LeNet5 CIFAR10
set(SOURCE_FILES
        ../src/datasets/cifar.cpp
        ../src/models/cnn/lenet5.cpp
)
add_executable(lenet5-cifar10 ${BASE_SOURCE_FILES} ${SOURCE_FILES} cnn/lenet5-cifar10.cpp)
target_link_libraries(lenet5-cifar10 "${TORCH_LIBRARIES}")
set_property(TARGET lenet5-cifar10 PROPERTY CXX_STANDARD 17)
target_include_directories(lenet5-cifar10 PRIVATE ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS})
target_link_libraries(lenet5-cifar10 ${TEST_LIBRARIES})

# LeNet5 CIFAR100
set(SOURCE_FILES
        ../src/datasets/cifar.cpp
        ../src/models/cnn/lenet5.cpp
)
add_executable(lenet5-cifar100 ${BASE_SOURCE_FILES} ${SOURCE_FILES} cnn/lenet5-cifar100.cpp)
target_link_libraries(lenet5-cifar100 "${TORCH_LIBRARIES}")
set_property(TARGET lenet5-cifar100 PROPERTY CXX_STANDARD 17)
target_include_directories(lenet5-cifar100 PRIVATE ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS})
target_link_libraries(lenet5-cifar100 ${TEST_LIBRARIES})



# AlexNet MNIST
set(SOURCE_FILES
        ../src/datasets/mnist.cpp
        ../src/models/cnn/alexnet.cpp
)
add_executable(alexnet-mnist ${BASE_SOURCE_FILES} ${SOURCE_FILES} cnn/alexnet-mnist.cpp)
target_link_libraries(alexnet-mnist "${TORCH_LIBRARIES}")
set_property(TARGET alexnet-mnist PROPERTY CXX_STANDARD 17)
target_include_directories(alexnet-mnist PRIVATE ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS})
target_link_libraries(alexnet-mnist ${TEST_LIBRARIES})
