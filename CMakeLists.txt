cmake_minimum_required(VERSION 3.1)
project(calibDLR11 VERSION 0.1)

#------------------------------------------------------------------------------
# Dependencies
#------------------------------------------------------------------------------
include(GNUInstallDirs)

find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc calib3d)
include_directories(${OpenCV_INCLUDE_DIRS})

OPTION(BUILD_TESTS "Enable testing" ON)

#------------------------------------------------------------------------------
# Build
#------------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

include_directories("include")

# Compiler 
if (WIN32)
    if (MSVC)
        add_compile_options(/W4)
    endif(MSVC)
else(WIN32)
    add_compile_options(-Wall)
endif(WIN32)

# set library and executable locations
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

aux_source_directory("src" LIBSRCALL)
add_library(${PROJECT_NAME} STATIC ${LIBSRCALL})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

if (BUILD_TESTS)
    enable_testing()
    link_libraries(${PROJECT_NAME})
    aux_source_directory("test" TMVTESTSRC)
    foreach (PERSRC ${TMVTESTSRC})
        string(REGEX REPLACE "test/(test_.*)\.cpp" "\\1" PERTGT ${PERSRC})
        add_executable(${PERTGT} ${PERSRC})
    endforeach (PERSRC)
endif(BUILD_TESTS)

#------------------------------------------------------------------------------
# Installation
#------------------------------------------------------------------------------
install(
    DIRECTORY
    "include/"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )

install(TARGETS ${PROJECT_NAME}
    RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
