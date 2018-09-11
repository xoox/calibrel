cmake_minimum_required(VERSION 3.1)
project(calibrel VERSION 0.2)

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

#-- Add an Option to toggle the generation of the API documentation
option(BUILD_DOCUMENTATION "Use Doxygen to create the HTML based API documentation" OFF)
if(BUILD_DOCUMENTATION)
    find_package(Doxygen)
    if (NOT DOXYGEN_FOUND)
    message(FATAL_ERROR
      "Doxygen is needed to build the documentation. Please install it correctly")
    endif()
    if(DOXYGEN_VERSION VERSION_LESS "1.8.0")
        message(WARNING "Doxygen >=1.8.0 is needed for better output.")
    endif()
    # - DOXYGEN_STRIP_FROM_INC_PATH corresponding to STRIP_FROM_INC_PATH in the doxy file
    set(DOXYGEN_STRIP_FROM_INC_PATH "\"${CMAKE_SOURCE_DIR}/include\"")
    #-- Configure the Template Doxyfile for our specific project
    configure_file(doc/Doxyfile.in
                 ${PROJECT_BINARY_DIR}/doc/Doxyfile  @ONLY IMMEDIATE)
    #-- Add a custom target to run Doxygen when ever the project is built
    add_custom_target (docs
                     COMMAND ${DOXYGEN_EXECUTABLE}
                     ${PROJECT_BINARY_DIR}/doc/Doxyfile
                     SOURCES ${PROJECT_BINARY_DIR}/doc/Doxyfile)
    # IF you do NOT want the documentation to be generated EVERY time you build the project
    # then leave out the 'ALL' keyword from the above command.
    install(DIRECTORY ${PROJECT_BINARY_DIR}/doc/html
        DESTINATION ${CMAKE_INSTALL_DOCDIR}
        COMPONENT doc)
endif(BUILD_DOCUMENTATION)
