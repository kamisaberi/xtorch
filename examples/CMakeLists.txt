project(examples)
list(APPEND CMAKE_PREFIX_PATH /home/kami/libs/c++/libtorch/)
find_package(Torch REQUIRED)
#find_package(TorchExtension REQUIRED)
find_package(LibLZMA REQUIRED)
find_package(LibArchive REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(BASE_SOURCE_FILES
        ../src/utils/extract.cpp
        ../src/utils/downloader.cpp
        ../src/utils/filesystem.cpp
        ../src/utils/md5.cpp
        ../src/utils/string.cpp
        ../src/media/image/processing.cpp
        ../src/media/opencv/images.cpp
#        ../src/datasets/mnist.cpp
#        ../src/models/cnn/lenet5.cpp
        ../src/definitions/transforms.cpp
        ../src/transforms/compose.cpp
        ../src/transforms/normalize.cpp
        ../src/transforms/resize.cpp
        ../src/transforms/lambda.cpp
        ../src/datasets/base/base.cpp
        ../src/datasets/definitions/functions.cpp
        ../src/models/base.cpp
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
        ${LibArchive_LIBRARIES}

)


######################################## APPLICATIONS ########################################



#--------------------------------------- CNN/AlexNet ---------------------------------------#

# AlexNet MNIST
set(SOURCE_FILES
        ../src/datasets/image-classification/mnist.cpp
        ../src/models/cnn/alexnet.cpp
)
add_executable(alexnet-mnist ${BASE_SOURCE_FILES} ${SOURCE_FILES} cnn/alexnet-mnist.cpp)
target_link_libraries(alexnet-mnist "${TORCH_LIBRARIES}")
set_property(TARGET alexnet-mnist PROPERTY CXX_STANDARD 17)
target_include_directories(alexnet-mnist PRIVATE ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS} ${LibArchive_INCLUDE_DIRS})
target_link_libraries(alexnet-mnist ${TEST_LIBRARIES})



# AlexNet FashionMNIST
set(SOURCE_FILES
        ../src/datasets/image-classification/mnist.cpp
        ../src/models/cnn/alexnet.cpp
)
add_executable(alexnet-fashionmnist ${BASE_SOURCE_FILES} ${SOURCE_FILES} cnn/alexnet-fashionmnist.cpp)
target_link_libraries(alexnet-fashionmnist "${TORCH_LIBRARIES}")
set_property(TARGET alexnet-fashionmnist PROPERTY CXX_STANDARD 17)
target_include_directories(alexnet-fashionmnist PRIVATE ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS} ${LibArchive_INCLUDE_DIRS})
target_link_libraries(alexnet-fashionmnist ${TEST_LIBRARIES})

# AlexNet CIFAR10
set(SOURCE_FILES
        ../src/datasets/image-classification/cifar.cpp
        ../src/models/cnn/alexnet.cpp
)
add_executable(alexnet-cifar10 ${BASE_SOURCE_FILES} ${SOURCE_FILES} cnn/alexnet-cifar10.cpp)
target_link_libraries(alexnet-cifar10 "${TORCH_LIBRARIES}")
set_property(TARGET alexnet-cifar10 PROPERTY CXX_STANDARD 17)
target_include_directories(alexnet-cifar10 PRIVATE ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS} ${LibArchive_INCLUDE_DIRS})
target_link_libraries(alexnet-cifar10 ${TEST_LIBRARIES})


# AlexNet CIFAR100
set(SOURCE_FILES
        ../src/datasets/image-classification/cifar.cpp
        ../src/models/cnn/alexnet.cpp
)
add_executable(alexnet-cifar100 ${BASE_SOURCE_FILES} ${SOURCE_FILES} cnn/alexnet-cifar100.cpp)
target_link_libraries(alexnet-cifar100 "${TORCH_LIBRARIES}")
set_property(TARGET alexnet-cifar100 PROPERTY CXX_STANDARD 17)
target_include_directories(alexnet-cifar100 PRIVATE ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS} ${LibArchive_INCLUDE_DIRS})
target_link_libraries(alexnet-cifar100 ${TEST_LIBRARIES})

# AlexNet Imagenette
set(SOURCE_FILES
        ../src/datasets/image-classification/imagenette.cpp
        ../src/models/cnn/alexnet.cpp
)
add_executable(alexnet-imagenette ${BASE_SOURCE_FILES} ${SOURCE_FILES} cnn/alexnet-imagenette.cpp)
target_link_libraries(alexnet-imagenette "${TORCH_LIBRARIES}")
set_property(TARGET alexnet-imagenette PROPERTY CXX_STANDARD 17)
target_include_directories(alexnet-imagenette PRIVATE ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS} ${LibArchive_INCLUDE_DIRS})
target_link_libraries(alexnet-imagenette ${TEST_LIBRARIES})


# AlexNet Food101
set(SOURCE_FILES
        ../src/datasets/image-classification/food.cpp
        ../src/models/cnn/alexnet.cpp
)
add_executable(alexnet-food101 ${BASE_SOURCE_FILES} ${SOURCE_FILES} cnn/alexnet-food101.cpp)
target_link_libraries(alexnet-food101 "${TORCH_LIBRARIES}")
set_property(TARGET alexnet-food101 PROPERTY CXX_STANDARD 17)
target_include_directories(alexnet-food101 PRIVATE ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS} ${LibArchive_INCLUDE_DIRS})
target_link_libraries(alexnet-food101 ${TEST_LIBRARIES})






#--------------------------------------- CNN/LeNet ---------------------------------------#

# LeNet5 MNIST
set(SOURCE_FILES
        ../src/datasets/image-classification/mnist.cpp
        ../src/models/cnn/lenet5.cpp

)
add_executable(lenet5-mnist ${BASE_SOURCE_FILES} ${SOURCE_FILES} cnn/lenet5-mnist.cpp)
target_link_libraries(lenet5-mnist "${TORCH_LIBRARIES}")
set_property(TARGET lenet5-mnist PROPERTY CXX_STANDARD 17)
target_include_directories(lenet5-mnist PRIVATE ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS} ${LibArchive_INCLUDE_DIRS})
target_link_libraries(lenet5-mnist ${TEST_LIBRARIES})

# LeNet5 FashionMNIST
set(SOURCE_FILES
        ../src/datasets/image-classification/mnist.cpp
        ../src/models/cnn/lenet5.cpp

)
add_executable(lenet5-fashionmnist ${BASE_SOURCE_FILES} ${SOURCE_FILES} cnn/lenet5-fashionmnist.cpp)
target_link_libraries(lenet5-fashionmnist "${TORCH_LIBRARIES}")
set_property(TARGET lenet5-fashionmnist PROPERTY CXX_STANDARD 17)
target_include_directories(lenet5-fashionmnist PRIVATE ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS} ${LibArchive_INCLUDE_DIRS})
target_link_libraries(lenet5-fashionmnist ${TEST_LIBRARIES})


# LeNet5 CIFAR10
set(SOURCE_FILES
        ../src/datasets/image-classification/cifar.cpp
        ../src/models/cnn/lenet5.cpp
)
add_executable(lenet5-cifar10 ${BASE_SOURCE_FILES} ${SOURCE_FILES} cnn/lenet5-cifar10.cpp)
target_link_libraries(lenet5-cifar10 "${TORCH_LIBRARIES}")
set_property(TARGET lenet5-cifar10 PROPERTY CXX_STANDARD 17)
target_include_directories(lenet5-cifar10 PRIVATE ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS} ${LibArchive_INCLUDE_DIRS})
target_link_libraries(lenet5-cifar10 ${TEST_LIBRARIES})

# LeNet5 CIFAR100
set(SOURCE_FILES
        ../src/datasets/image-classification/cifar.cpp
        ../src/models/cnn/lenet5.cpp
)
add_executable(lenet5-cifar100 ${BASE_SOURCE_FILES} ${SOURCE_FILES} cnn/lenet5-cifar100.cpp)
target_link_libraries(lenet5-cifar100 "${TORCH_LIBRARIES}")
set_property(TARGET lenet5-cifar100 PROPERTY CXX_STANDARD 17)
target_include_directories(lenet5-cifar100 PRIVATE ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS} ${LibArchive_INCLUDE_DIRS})
target_link_libraries(lenet5-cifar100 ${TEST_LIBRARIES})


# LeNet5 Imagenette
set(SOURCE_FILES
        ../src/datasets/image-classification/imagenette.cpp
        ../src/models/cnn/lenet5.cpp
)
add_executable(lenet5-imagenette ${BASE_SOURCE_FILES} ${SOURCE_FILES} cnn/lenet5-imagenette.cpp)
target_link_libraries(lenet5-imagenette "${TORCH_LIBRARIES}")
set_property(TARGET lenet5-imagenette PROPERTY CXX_STANDARD 17)
target_include_directories(lenet5-imagenette PRIVATE ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS} ${LibArchive_INCLUDE_DIRS})
target_link_libraries(lenet5-imagenette ${TEST_LIBRARIES})


# LeNet5 Food101
set(SOURCE_FILES
        ../src/datasets/image-classification/food.cpp
        ../src/models/cnn/lenet5.cpp
)
add_executable(lenet5-food101 ${BASE_SOURCE_FILES} ${SOURCE_FILES} cnn/lenet5-food101.cpp)
target_link_libraries(lenet5-food101 "${TORCH_LIBRARIES}")
set_property(TARGET lenet5-food101 PROPERTY CXX_STANDARD 17)
target_include_directories(lenet5-food101 PRIVATE ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS} ${LibArchive_INCLUDE_DIRS})
target_link_libraries(lenet5-food101 ${TEST_LIBRARIES})







#--------------------------------------- CNN/ResNet ---------------------------------------#

# ResNet MNIST
set(SOURCE_FILES
        ../src/datasets/image-classification/mnist.cpp
        ../src/models/cnn/resnet.cpp
)
add_executable(resnet-mnist ${BASE_SOURCE_FILES} ${SOURCE_FILES} cnn/resnet-mnist.cpp)
target_link_libraries(resnet-mnist "${TORCH_LIBRARIES}")
set_property(TARGET resnet-mnist PROPERTY CXX_STANDARD 17)
target_include_directories(resnet-mnist PRIVATE ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS} ${LibArchive_INCLUDE_DIRS})
target_link_libraries(resnet-mnist ${TEST_LIBRARIES})


# ResNet FashionMNIST
set(SOURCE_FILES
        ../src/datasets/image-classification/mnist.cpp
        ../src/models/cnn/resnet.cpp
)
add_executable(resnet-fashionmnist ${BASE_SOURCE_FILES} ${SOURCE_FILES} cnn/resnet-fashionmnist.cpp)
target_link_libraries(resnet-fashionmnist "${TORCH_LIBRARIES}")
set_property(TARGET resnet-fashionmnist PROPERTY CXX_STANDARD 17)
target_include_directories(resnet-fashionmnist PRIVATE ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS} ${LibArchive_INCLUDE_DIRS})
target_link_libraries(resnet-fashionmnist ${TEST_LIBRARIES})

# ResNet CIFAR10
set(SOURCE_FILES
        ../src/datasets/image-classification/cifar.cpp
        ../src/models/cnn/resnet.cpp
)
add_executable(resnet-cifar10 ${BASE_SOURCE_FILES} ${SOURCE_FILES} cnn/resnet-cifar10.cpp)
target_link_libraries(resnet-cifar10 "${TORCH_LIBRARIES}")
set_property(TARGET resnet-cifar10 PROPERTY CXX_STANDARD 17)
target_include_directories(resnet-cifar10 PRIVATE ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS} ${LibArchive_INCLUDE_DIRS})
target_link_libraries(resnet-cifar10 ${TEST_LIBRARIES})


# ResNet CIFAR100
set(SOURCE_FILES
        ../src/datasets/image-classification/cifar.cpp
        ../src/models/cnn/resnet.cpp
)
add_executable(resnet-cifar100 ${BASE_SOURCE_FILES} ${SOURCE_FILES} cnn/resnet-cifar100.cpp)
target_link_libraries(resnet-cifar100 "${TORCH_LIBRARIES}")
set_property(TARGET resnet-cifar100 PROPERTY CXX_STANDARD 17)
target_include_directories(resnet-cifar100 PRIVATE ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS} ${LibArchive_INCLUDE_DIRS})
target_link_libraries(resnet-cifar100 ${TEST_LIBRARIES})

# ResNet Imagenette
set(SOURCE_FILES
        ../src/datasets/image-classification/imagenette.cpp
        ../src/models/cnn/resnet.cpp
)
add_executable(resnet-imagenette ${BASE_SOURCE_FILES} ${SOURCE_FILES} cnn/resnet-imagenette.cpp)
target_link_libraries(resnet-imagenette "${TORCH_LIBRARIES}")
set_property(TARGET resnet-imagenette PROPERTY CXX_STANDARD 17)
target_include_directories(resnet-imagenette PRIVATE ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS} ${LibArchive_INCLUDE_DIRS})
target_link_libraries(resnet-imagenette ${TEST_LIBRARIES})


# ResNet Food101
set(SOURCE_FILES
        ../src/datasets/image-classification/food.cpp
        ../src/models/cnn/resnet.cpp
)
add_executable(resnet-food101 ${BASE_SOURCE_FILES} ${SOURCE_FILES} cnn/resnet-food101.cpp)
target_link_libraries(resnet-food101 "${TORCH_LIBRARIES}")
set_property(TARGET resnet-food101 PROPERTY CXX_STANDARD 17)
target_include_directories(resnet-food101 PRIVATE ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS} ${LibArchive_INCLUDE_DIRS})
target_link_libraries(resnet-food101 ${TEST_LIBRARIES})


#--------------------------------------- CNN/VggNet ---------------------------------------#

# VggNet MNIST
set(SOURCE_FILES
        ../src/datasets/image-classification/mnist.cpp
        ../src/models/cnn/vggnet.cpp
)
add_executable(vggnet16-mnist ${BASE_SOURCE_FILES} ${SOURCE_FILES} cnn/vggnet16-mnist.cpp)
target_link_libraries(vggnet16-mnist "${TORCH_LIBRARIES}")
set_property(TARGET vggnet16-mnist PROPERTY CXX_STANDARD 17)
target_include_directories(vggnet16-mnist PRIVATE ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS} ${LibArchive_INCLUDE_DIRS})
target_link_libraries(vggnet16-mnist ${TEST_LIBRARIES})



# VggNet FashionMNIST
set(SOURCE_FILES
        ../src/datasets/image-classification/mnist.cpp
        ../src/models/cnn/vggnet.cpp
)
add_executable(vggnet16-fashionmnist ${BASE_SOURCE_FILES} ${SOURCE_FILES} cnn/vggnet16-fashionmnist.cpp)
target_link_libraries(vggnet16-fashionmnist "${TORCH_LIBRARIES}")
set_property(TARGET vggnet16-fashionmnist PROPERTY CXX_STANDARD 17)
target_include_directories(vggnet16-fashionmnist PRIVATE ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS} ${LibArchive_INCLUDE_DIRS})
target_link_libraries(vggnet16-fashionmnist ${TEST_LIBRARIES})

# VggNet CIFAR10
set(SOURCE_FILES
        ../src/datasets/image-classification/cifar.cpp
        ../src/models/cnn/vggnet.cpp
)
add_executable(vggnet16-cifar10 ${BASE_SOURCE_FILES} ${SOURCE_FILES} cnn/vggnet16-cifar10.cpp)
target_link_libraries(vggnet16-cifar10 "${TORCH_LIBRARIES}")
set_property(TARGET vggnet16-cifar10 PROPERTY CXX_STANDARD 17)
target_include_directories(vggnet16-cifar10 PRIVATE ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS} ${LibArchive_INCLUDE_DIRS})
target_link_libraries(vggnet16-cifar10 ${TEST_LIBRARIES})


# VggNet CIFAR100
set(SOURCE_FILES
        ../src/datasets/image-classification/cifar.cpp
        ../src/models/cnn/vggnet.cpp
)
add_executable(vggnet16-cifar100 ${BASE_SOURCE_FILES} ${SOURCE_FILES} cnn/vggnet16-cifar100.cpp)
target_link_libraries(vggnet16-cifar100 "${TORCH_LIBRARIES}")
set_property(TARGET vggnet16-cifar100 PROPERTY CXX_STANDARD 17)
target_include_directories(vggnet16-cifar100 PRIVATE ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS} ${LibArchive_INCLUDE_DIRS})
target_link_libraries(vggnet16-cifar100 ${TEST_LIBRARIES})

# VggNet Imagenette
set(SOURCE_FILES
        ../src/datasets/image-classification/imagenette.cpp
        ../src/models/cnn/vggnet.cpp
)
add_executable(vggnet16-imagenette ${BASE_SOURCE_FILES} ${SOURCE_FILES} cnn/vggnet16-imagenette.cpp)
target_link_libraries(vggnet16-imagenette "${TORCH_LIBRARIES}")
set_property(TARGET vggnet16-imagenette PROPERTY CXX_STANDARD 17)
target_include_directories(vggnet16-imagenette PRIVATE ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS} ${LibArchive_INCLUDE_DIRS})
target_link_libraries(vggnet16-imagenette ${TEST_LIBRARIES})

# VggNet Food101
set(SOURCE_FILES
        ../src/datasets/image-classification/food.cpp
        ../src/models/cnn/vggnet.cpp
)
add_executable(vggnet16-food101 ${BASE_SOURCE_FILES} ${SOURCE_FILES} cnn/vggnet16-food101.cpp)
target_link_libraries(vggnet16-food101 "${TORCH_LIBRARIES}")
set_property(TARGET vggnet16-food101 PROPERTY CXX_STANDARD 17)
target_include_directories(vggnet16-food101 PRIVATE ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS} ${LibArchive_INCLUDE_DIRS})
target_link_libraries(vggnet16-food101 ${TEST_LIBRARIES})



#--------------------------------------- Trainer ---------------------------------------#




#file(GLOB_RECURSE UTILITY_FILES CONFIGURE_DEPENDS ../src/utils/*.cpp)
#file(GLOB_RECURSE MEDIA_FILES CONFIGURE_DEPENDS ../src/media/*.cpp)
#file(GLOB_RECURSE DATASET_FILES CONFIGURE_DEPENDS ../src/datasets/*.cpp)
#file(GLOB_RECURSE DEFINITION_FILES CONFIGURE_DEPENDS ../src/definitions/*.cpp)
#file(GLOB_RECURSE MODEL_FILES CONFIGURE_DEPENDS ../src/models/*.cpp)
#file(GLOB_RECURSE MODEL_CNN_FILES CONFIGURE_DEPENDS ../src/models/cnn/*.cpp)
#file(GLOB_RECURSE DATALOADER_FILES CONFIGURE_DEPENDS ../src/data-loaders/*.tpp)
#file(GLOB_RECURSE TRANSFORM_FILES CONFIGURE_DEPENDS ../src/transforms/*.cpp)
## Combine all source files into one variable for easy access
#set(LIBRARY_SOURCE_FILES ${UTILITY_FILES} ${MEDIA_FILES} ${DATASET_FILIES} ${DEFINITION_FILES} ${MODEL_FILES}  ${DATALOADER_FILES} ${TRANSFORM_FILES} )


# Trainer Simple
set(SOURCE_FILES ${LIBRARY_SOURCE_FILES}
        ../src/datasets/image-classification/mnist.cpp
        ../src/models/cnn/lenet5.cpp
        ../src/trainers/trainer.cpp
        ../src/datasets/base/base.cpp

        ../src/data-loaders/data-loader.cpp

)
add_executable(trainer-simple ${BASE_SOURCE_FILES} ${SOURCE_FILES} trainer/simple.cpp)
target_link_libraries(trainer-simple "${TORCH_LIBRARIES}")
set_property(TARGET trainer-simple PROPERTY CXX_STANDARD 17)
target_include_directories(trainer-simple PRIVATE ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS} ${LibArchive_INCLUDE_DIRS})
target_link_libraries(trainer-simple ${TEST_LIBRARIES})
