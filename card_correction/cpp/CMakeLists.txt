cmake_minimum_required(VERSION 3.16)
project(card_correction_dll)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Windows设置
if(WIN32)
    add_definitions(-DCARD_CORRECTION_EXPORTS)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

# 查找OpenCV
find_package(OpenCV REQUIRED)

# 源文件
set(SOURCES
    card_correction_dll.cpp
    predictor.cpp
    utils.cpp
)

# 头文件
set(HEADERS
    card_correction_dll.h
    predictor.h
    utils.h
)

# 创建共享库(DLL)
add_library(card_correction SHARED ${SOURCES} ${HEADERS})

# 链接OpenCV
target_link_libraries(card_correction ${OpenCV_LIBS})

# 设置输出名称
set_target_properties(card_correction PROPERTIES
    OUTPUT_NAME "card_correction"
    PREFIX ""
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# 创建测试可执行文件
add_executable(test_dll test_dll.cpp)
target_link_libraries(test_dll card_correction ${OpenCV_LIBS})
set_target_properties(test_dll PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# 安装目标
install(TARGETS card_correction
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES card_correction_dll.h DESTINATION include)

# 打印配置信息
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
