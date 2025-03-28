cmake_minimum_required(VERSION 3.28 FATAL_ERROR)


# List Of All CPP Files By Category
file(GLOB_RECURSE UTILITY_FILES CONFIGURE_DEPENDS ../src/utils/*.cpp)
file(GLOB_RECURSE MEDIA_FILES CONFIGURE_DEPENDS ../src/media/*.cpp)
file(GLOB_RECURSE DATASET_FILES CONFIGURE_DEPENDS ../src/datasets/*.cpp)
file(GLOB_RECURSE DEFINITION_FILES CONFIGURE_DEPENDS ../src/definitions/*.cpp)
file(GLOB_RECURSE MODEL_FILES CONFIGURE_DEPENDS ../src/models/*.cpp)
file(GLOB_RECURSE MODEL_CNN_FILES CONFIGURE_DEPENDS ../src/models/cnn/*.cpp)



####################################### TEST #######################################


project(Test)
list(APPEND CMAKE_PREFIX_PATH /home/kami/libs/c++/libtorch/)
find_package(Torch REQUIRED)
find_package(TorchExtension REQUIRED)
find_package(LibLZMA REQUIRED)
find_package(LibArchive REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(TEST_SOURCE_FILES ${UTILITY_FILES} ${MEDIA_FILES} ../src/datasets/mnist.cpp test.cpp)

add_executable(${PROJECT_NAME} ${TEST_SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")
#
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

target_include_directories(${PROJECT_NAME} PRIVATE ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS} ${LibArchive_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME}
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


####################################### MD #######################################


project(md)
list(APPEND CMAKE_PREFIX_PATH /home/kami/libs/c++/libtorch/)
find_package(Torch REQUIRED)
find_package(TorchExtension REQUIRED)
find_package(LibLZMA REQUIRED)
find_package(LibArchive REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)


set(TEST_SOURCE_FILES ../src/utils/md5.cpp md.cpp)

add_executable(${PROJECT_NAME} ${TEST_SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")
#
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

target_include_directories(${PROJECT_NAME} PRIVATE ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS} ${LibArchive_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME}
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


####################################### MD2 #######################################


project(md2)
list(APPEND CMAKE_PREFIX_PATH /home/kami/libs/c++/libtorch/)
find_package(Torch REQUIRED)
find_package(TorchExtension REQUIRED)
find_package(LibLZMA REQUIRED)
find_package(LibArchive REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(TEST_SOURCE_FILES ../src/utils/md5.cpp md2.cpp)

add_executable(${PROJECT_NAME} ${TEST_SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")
#
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

target_include_directories(${PROJECT_NAME} PRIVATE ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS} ${LibArchive_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME}
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

####################################### XZ #######################################



project(xz)
list(APPEND CMAKE_PREFIX_PATH /home/kami/libs/c++/libtorch/)
find_package(Torch REQUIRED)
find_package(TorchExtension REQUIRED)
find_package(LibLZMA REQUIRED)
find_package(LibArchive REQUIRED)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(TEST_SOURCE_FILES xz.cpp)

add_executable(${PROJECT_NAME} ${TEST_SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")
#
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

target_include_directories(${PROJECT_NAME} PRIVATE ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS} ${LibArchive_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME}
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


####################################### LETEST #######################################


project(letest)
list(APPEND CMAKE_PREFIX_PATH /home/kami/libs/c++/libtorch/)
find_package(Torch REQUIRED)
find_package(TorchExtension REQUIRED)
find_package(LibLZMA REQUIRED)
find_package(LibArchive REQUIRED)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(TEST_SOURCE_FILES ${UTILITY_FILES} ${MEDIA_FILES} ${DEFINITION_FILES} ../src/datasets/mnist.cpp ../src/models/cnn/lenet5.cpp letest.cpp)

add_executable(${PROJECT_NAME} ${TEST_SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")
#
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

target_include_directories(${PROJECT_NAME} PRIVATE ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS} ${LibArchive_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME}
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

####################################### MNT #######################################


project(mnt)
list(APPEND CMAKE_PREFIX_PATH /home/kami/libs/c++/libtorch/)
find_package(Torch REQUIRED)
find_package(TorchExtension REQUIRED)
find_package(LibLZMA REQUIRED)
find_package(LibArchive REQUIRED)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(TEST_SOURCE_FILES ${UTILITY_FILES} ${MEDIA_FILES} ${DEFINITION_FILES} ${DATASET_FILES} ../src/models/cnn/lenet5.cpp mnt.cpp)

add_executable(${PROJECT_NAME} ${TEST_SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")
#
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

target_include_directories(${PROJECT_NAME} PRIVATE ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS} ${LibArchive_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME}
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

set(TEST_SOURCE_FILES ${UTILITY_FILES} ${MEDIA_FILES} ${DEFINITION_FILES} ${DATASET_FILES} ${MODEL_FILES} tgz.cpp)

add_executable(${PROJECT_NAME} ${TEST_SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")
#
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

target_include_directories(${PROJECT_NAME} PRIVATE ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS} ${LibArchive_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME}
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


set(TEST_SOURCE_FILES ${UTILITY_FILES} ${MEDIA_FILES} ${DEFINITION_FILES} ${DATASET_FILES} ${MODEL_FILES} imgnt.cpp)

add_executable(${PROJECT_NAME} ${TEST_SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")
#
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

include_directories(/usr/include)

target_include_directories(${PROJECT_NAME} PRIVATE ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS} ${LibArchive_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME}
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

set(TEST_SOURCE_FILES ${UTILITY_FILES} ${MEDIA_FILES} ${DEFINITION_FILES} ${DATASET_FILES} ${MODEL_FILES} vdtest.cpp)

add_executable(${PROJECT_NAME} ${TEST_SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")
#
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

include_directories(/usr/include)

target_include_directories(${PROJECT_NAME} PRIVATE ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS} ${LibArchive_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME}
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

set(TEST_SOURCE_FILES ${UTILITY_FILES} ${MEDIA_FILES} ${DEFINITION_FILES} ${DATASET_FILES} ${MODEL_FILES} cntfiles.cpp)

add_executable(${PROJECT_NAME} ${TEST_SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")
#
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

include_directories(/usr/include)

target_include_directories(${PROJECT_NAME} PRIVATE ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS} ${LibArchive_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME}
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



####################################### IMGRESIZE #######################################

project(imgresize)
list(APPEND CMAKE_PREFIX_PATH /home/kami/libs/c++/libtorch/)
find_package(Torch REQUIRED)
find_package(TorchExtension REQUIRED)
find_package(OpenCV REQUIRED)
find_package(LibLZMA REQUIRED)
find_package(LibArchive REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(TEST_SOURCE_FILES ${UTILITY_FILES} ${MEDIA_FILES} ${DEFINITION_FILES} ${DATASET_FILES} ${MODEL_FILES} imgresize.cpp)

add_executable(${PROJECT_NAME} ${TEST_SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")
#
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

include_directories(/usr/include)

target_include_directories(${PROJECT_NAME} PRIVATE ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS} ${LibArchive_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME}
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









####################################### DLTEST3 #######################################

project(dltest3)
list(APPEND CMAKE_PREFIX_PATH /home/kami/libs/c++/libtorch/)
find_package(Torch REQUIRED)
find_package(TorchExtension REQUIRED)
find_package(OpenCV REQUIRED)
find_package(LibLZMA REQUIRED)
find_package(LibArchive REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(TEST_SOURCE_FILES ${UTILITY_FILES} ${MEDIA_FILES} ${DEFINITION_FILES} ${DATASET_FILES} ${MODEL_FILES} dltest3.cpp)

add_executable(${PROJECT_NAME} ${TEST_SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")
#
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

include_directories(/usr/include)

target_include_directories(${PROJECT_NAME} PRIVATE ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS} ${LibArchive_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME}
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



####################################### DLTEST4 #######################################

project(dltest4)
list(APPEND CMAKE_PREFIX_PATH /home/kami/libs/c++/libtorch/)
find_package(Torch REQUIRED)
find_package(TorchExtension REQUIRED)
find_package(OpenCV REQUIRED)
find_package(LibLZMA REQUIRED)
find_package(LibArchive REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(TEST_SOURCE_FILES ${UTILITY_FILES} ${MEDIA_FILES} ${DEFINITION_FILES} ${DATASET_FILES} ${MODEL_FILES} dltest4.cpp)

add_executable(${PROJECT_NAME} ${TEST_SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")
#
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

include_directories(/usr/include)

target_include_directories(${PROJECT_NAME} PRIVATE ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS} ${LibArchive_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME}
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



####################################### TRTEST #######################################

project(trtest)
list(APPEND CMAKE_PREFIX_PATH /home/kami/libs/c++/libtorch/)
find_package(Torch REQUIRED)
find_package(TorchExtension REQUIRED)
find_package(OpenCV REQUIRED)
find_package(LibLZMA REQUIRED)
find_package(LibArchive REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(TEST_SOURCE_FILES ${UTILITY_FILES} ${MEDIA_FILES} ${DEFINITION_FILES} ${DATASET_FILES} ${MODEL_FILES} trtest.cpp)

add_executable(${PROJECT_NAME} ${TEST_SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")
#
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

include_directories(/usr/include)

target_include_directories(${PROJECT_NAME} PRIVATE ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS} ${LibArchive_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME}
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




####################################### TRTEST2 #######################################

project(trtest2)
list(APPEND CMAKE_PREFIX_PATH /home/kami/libs/c++/libtorch/)
find_package(Torch REQUIRED)
find_package(TorchExtension REQUIRED)
find_package(OpenCV REQUIRED)
find_package(LibLZMA REQUIRED)
find_package(LibArchive REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(TEST_SOURCE_FILES ${UTILITY_FILES} ${MEDIA_FILES} ${DEFINITION_FILES} ${DATASET_FILES} ${MODEL_FILES} trtest2.cpp)

add_executable(${PROJECT_NAME} ${TEST_SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")
#
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

include_directories(/usr/include)

target_include_directories(${PROJECT_NAME} PRIVATE ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS} ${LibArchive_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME}
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
















