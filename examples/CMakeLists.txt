if(ANDROID)
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pie")
    get_filename_component(OPENCV_ROOT_DIR "/Users/keyong/Documents/gits/opencv.git/android_build_ndk14b/OpenCV-android-sdk/sdk/native" ABSOLUTE)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src)
    include_directories(${CMAKE_CURRENT_BINARY_DIR}/../src)
    include_directories(${OPENCV_ROOT_DIR}/jni/include)

    macro(ADD_OPENCV_STATIC_LIB libxxx)
        add_library(${libxxx} STATIC IMPORTED )
        set_target_properties(${libxxx} PROPERTIES IMPORTED_LOCATION
                "${OPENCV_ROOT_DIR}/staticlibs/${ANDROID_ABI}/${libxxx}.a")
    endmacro()

    ADD_OPENCV_STATIC_LIB(libopencv_bgsegm)
    ADD_OPENCV_STATIC_LIB(libopencv_bioinspired)
    ADD_OPENCV_STATIC_LIB(libopencv_calib3d)
    ADD_OPENCV_STATIC_LIB(libopencv_ccalib)
    ADD_OPENCV_STATIC_LIB(libopencv_imgcodecs)
    ADD_OPENCV_STATIC_LIB(libopencv_imgproc)
    ADD_OPENCV_STATIC_LIB(libopencv_core)
    ADD_OPENCV_STATIC_LIB(libopencv_datasets)
    ADD_OPENCV_STATIC_LIB(libopencv_dnn)
    ADD_OPENCV_STATIC_LIB(libopencv_dpm)
    ADD_OPENCV_STATIC_LIB(libopencv_face)
    ADD_OPENCV_STATIC_LIB(libopencv_features2d)
    ADD_OPENCV_STATIC_LIB(libopencv_flann)
    ADD_OPENCV_STATIC_LIB(libopencv_fuzzy)
    ADD_OPENCV_STATIC_LIB(libopencv_highgui)
    ADD_OPENCV_STATIC_LIB(libopencv_line_descriptor)
    ADD_OPENCV_STATIC_LIB(libopencv_ml)
    ADD_OPENCV_STATIC_LIB(libopencv_objdetect)
    ADD_OPENCV_STATIC_LIB(libopencv_optflow)
    ADD_OPENCV_STATIC_LIB(libopencv_phase_unwrapping)
    ADD_OPENCV_STATIC_LIB(libopencv_photo)
    ADD_OPENCV_STATIC_LIB(libopencv_plot)
    ADD_OPENCV_STATIC_LIB(libopencv_reg)
    ADD_OPENCV_STATIC_LIB(libopencv_rgbd)
    ADD_OPENCV_STATIC_LIB(libopencv_saliency)
    ADD_OPENCV_STATIC_LIB(libopencv_shape)
    ADD_OPENCV_STATIC_LIB(libopencv_stereo)
    ADD_OPENCV_STATIC_LIB(libopencv_stitching)
    ADD_OPENCV_STATIC_LIB(libopencv_structured_light)
    ADD_OPENCV_STATIC_LIB(libopencv_superres)
    ADD_OPENCV_STATIC_LIB(libopencv_surface_matching)
    ADD_OPENCV_STATIC_LIB(libopencv_text)
    ADD_OPENCV_STATIC_LIB(libopencv_tracking)
    ADD_OPENCV_STATIC_LIB(libopencv_video)
    ADD_OPENCV_STATIC_LIB(libopencv_videoio)
    ADD_OPENCV_STATIC_LIB(libopencv_videostab)
    ADD_OPENCV_STATIC_LIB(libopencv_xfeatures2d)
    ADD_OPENCV_STATIC_LIB(libopencv_ximgproc)
    ADD_OPENCV_STATIC_LIB(libopencv_xobjdetect)
    ADD_OPENCV_STATIC_LIB(libopencv_xphoto)

    #opencv 3rdparty
    macro(ADD_OPENCV_3RDPARTY_LIB libxxx)
        add_library(${libxxx} STATIC IMPORTED )
        set_target_properties(${libxxx} PROPERTIES
                IMPORTED_LOCATION "${OPENCV_ROOT_DIR}/3rdparty/libs/${ANDROID_ABI}/${libxxx}.a")
    endmacro()

    ADD_OPENCV_3RDPARTY_LIB(libIlmImf)
    ADD_OPENCV_3RDPARTY_LIB(liblibjasper)
    ADD_OPENCV_3RDPARTY_LIB(liblibjpeg-turbo)
    ADD_OPENCV_3RDPARTY_LIB(liblibpng)
    ADD_OPENCV_3RDPARTY_LIB(liblibtiff)
    ADD_OPENCV_3RDPARTY_LIB(liblibwebp)
    ADD_OPENCV_3RDPARTY_LIB(libtegra_hal)
    ADD_OPENCV_3RDPARTY_LIB(libcpufeatures)
    ADD_OPENCV_3RDPARTY_LIB(libtbb)
    ADD_OPENCV_3RDPARTY_LIB(libquirc)

    set(OpenCV_LIBS   libopencv_calib3d
            #libopencv_ccalib
            libopencv_imgcodecs
            libopencv_imgproc
            libopencv_core
            #libopencv_datasets
            #libopencv_dpm
            #libopencv_face
            libopencv_features2d
            libopencv_flann
            #libopencv_fuzzy
            #libopencv_highgui
            #libopencv_line_descriptor
            #libopencv_ml
            libopencv_objdetect
            #libopencv_optflow
            #libopencv_phase_unwrapping
            libopencv_photo
            #libopencv_plot
            #libopencv_reg
            #libopencv_rgbd
            #libopencv_saliency
            #libopencv_shape
            #libopencv_stereo
            libopencv_stitching
            #libopencv_structured_light
            libopencv_superres
            #libopencv_surface_matching
            #libopencv_text
            #libopencv_tracking
            libopencv_video
            libopencv_videoio
            libopencv_videostab
            #libopencv_xfeatures2d
            #libopencv_ximgproc
            #libopencv_xobjdetect
            #libopencv_xphoto
            #libopencv_dnn
            #opencv 3rdparty
            libIlmImf
            liblibjasper
            liblibjpeg-turbo
            liblibpng
            liblibtiff
            liblibwebp
            libtegra_hal
            libcpufeatures
            libtbb
            log
            z
            )
else()
    if (IOS)
        #search opencv2.framework in DMi_RnD folder
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -F/Users/keyong/Documents/gits/DMi_RnD")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -framework opencv2")
    else ()
        find_package(OpenCV QUIET COMPONENTS core highgui imgproc imgcodecs)
        if (NOT OpenCV_FOUND)
            find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc)
        endif ()
    endif()
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../src)

set(NCNN_EXAMPLE_LINK_LIBRARIES ncnn ${OpenCV_LIBS})
if(NCNN_VULKAN)
    list(APPEND NCNN_EXAMPLE_LINK_LIBRARIES ${Vulkan_LIBRARY})
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIE -std=c++11 -mfpu=neon")

#add_executable(squeezenet squeezenet.cpp)
#target_link_libraries(squeezenet ${NCNN_EXAMPLE_LINK_LIBRARIES})
#
#add_executable(fasterrcnn fasterrcnn.cpp)
#target_link_libraries(fasterrcnn ${NCNN_EXAMPLE_LINK_LIBRARIES})
#
#add_executable(rfcn rfcn.cpp)
#target_link_libraries(rfcn ${NCNN_EXAMPLE_LINK_LIBRARIES})
#
#add_executable(yolov2 yolov2.cpp)
#target_link_libraries(yolov2 ${NCNN_EXAMPLE_LINK_LIBRARIES})
#
#add_executable(yolov3 yolov3.cpp)
#target_link_libraries(yolov3 ${NCNN_EXAMPLE_LINK_LIBRARIES})
#
#add_executable(mobilenetv2ssdlite2 mobilenetv2ssdlite2.cpp)
#target_link_libraries(mobilenetv2ssdlite2 ${NCNN_EXAMPLE_LINK_LIBRARIES})
#
#add_executable(mobilenetssd mobilenetssd.cpp)
#target_link_libraries(mobilenetssd ${NCNN_EXAMPLE_LINK_LIBRARIES})
#
#add_executable(squeezenetssd squeezenetssd.cpp)
#target_link_libraries(squeezenetssd ${NCNN_EXAMPLE_LINK_LIBRARIES})
#
#add_executable(shufflenetv2 shufflenetv2.cpp)
#target_link_libraries(shufflenetv2 ${NCNN_EXAMPLE_LINK_LIBRARIES})

add_executable(mobilenetv2ssdlite2 mbv2ssdlite.cpp)
target_link_libraries(mobilenetv2ssdlite2 ${NCNN_EXAMPLE_LINK_LIBRARIES})

