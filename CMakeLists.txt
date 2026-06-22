cmake_minimum_required(VERSION 3.5)

project(YOLOAnnotator VERSION 0.1 LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

# 启用自动生成moc/uic/rcc（Qt必备）
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

# C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. 查找Qt依赖（确保包含 Sql 和 Charts）
find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Core Gui Widgets Sql Charts)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Core Gui Widgets Sql Charts)

# 2. 统一定义所有源文件
set(PROJECT_SOURCES
    main.cpp
    mainwindow.cpp
    mainwindow.h
    mainwindow.ui
    canvaswidget.cpp
    canvaswidget.h
    yoloutils.cpp
    yoloutils.h
    Annotation.h
    quantdialog.cpp  # 新增
    quantdialog.h
    onnxinferenceengine.cpp
    onnxinferenceengine.h
    qt_quant_infer_bridge.py
    qt_train_bridge.py
)

# 3. 生成可执行文件（移除了 .py 脚本和重复文件）
if(${QT_VERSION_MAJOR} GREATER_EQUAL 6)
    qt_add_executable(YOLOAnnotator
        MANUAL_FINALIZATION
        ${PROJECT_SOURCES}
    )
else()
    if(ANDROID)
        add_library(YOLOAnnotator SHARED
            ${PROJECT_SOURCES}
        )
    else()
        add_executable(YOLOAnnotator
            ${PROJECT_SOURCES}
        )
    endif()
endif()

# 4. 链接Qt库（关键：必须在这里添加 Sql 和 Charts 才能调用数据库和图表）
target_link_libraries(YOLOAnnotator PRIVATE
    Qt${QT_VERSION_MAJOR}::Core
    Qt${QT_VERSION_MAJOR}::Gui
    Qt${QT_VERSION_MAJOR}::Widgets
    Qt${QT_VERSION_MAJOR}::Sql
    Qt${QT_VERSION_MAJOR}::Charts
)

# Optional native ONNX Runtime backend. The application still builds without it
# and displays a configuration hint when C++ inference is selected.
set(ONNXRUNTIME_ROOT "" CACHE PATH "ONNX Runtime installation root")
if(ONNXRUNTIME_ROOT)
    find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
        PATHS "${ONNXRUNTIME_ROOT}/include" NO_DEFAULT_PATH)
    find_library(ONNXRUNTIME_LIBRARY onnxruntime
        PATHS "${ONNXRUNTIME_ROOT}/lib" NO_DEFAULT_PATH)
    if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIBRARY)
        target_include_directories(YOLOAnnotator PRIVATE "${ONNXRUNTIME_INCLUDE_DIR}")
        target_link_libraries(YOLOAnnotator PRIVATE "${ONNXRUNTIME_LIBRARY}")
        target_compile_definitions(YOLOAnnotator PRIVATE YOLO_WITH_ONNXRUNTIME=1)
        message(STATUS "ONNX Runtime C++ inference enabled")
        if(WIN32 AND EXISTS "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll")
            add_custom_command(TARGET YOLOAnnotator POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll" "$<TARGET_FILE_DIR:YOLOAnnotator>")
        endif()
    else()
        message(WARNING "ONNXRUNTIME_ROOT is set but headers/library were not found; C++ inference disabled")
    endif()
endif()

foreach(BRIDGE_SCRIPT qt_train_bridge.py qt_quant_infer_bridge.py)
    add_custom_command(TARGET YOLOAnnotator POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/${BRIDGE_SCRIPT}" "$<TARGET_FILE_DIR:YOLOAnnotator>/${BRIDGE_SCRIPT}")
endforeach()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/yolo26n.pt")
    add_custom_command(TARGET YOLOAnnotator POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/yolo26n.pt" "$<TARGET_FILE_DIR:YOLOAnnotator>/yolo26n.pt")
endif()

# Qt6额外配置
if(${QT_VERSION} VERSION_GREATER_EQUAL 6.1)
    target_compile_definitions(YOLOAnnotator PRIVATE QT_DEPRECATED_WARNINGS)
endif()

# 最终化配置（Qt6）
if(QT_VERSION_MAJOR EQUAL 6)
    qt_finalize_executable(YOLOAnnotator)
endif()
