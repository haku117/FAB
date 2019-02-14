cmake_minimum_required (VERSION 3.0)

macro(use_cxx11)
  if (CMAKE_VERSION VERSION_LESS "3.1")
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    endif ()
  else ()
    set (CMAKE_CXX_STANDARD 11)
  endif ()
endmacro(use_cxx11)

project(lr C CXX)

use_cxx11()

# -------- Boost --------
#find_package(Boost REQUIRED COMPONENTS filesystem system program_options)
#include_directories(${Boost_INCLUDE_DIRS}) 

# -------- Python interface --------
#find_package(PythonLibs)
#find_package(SWIG)

# -------- Protobuf --------
#find_package(HAVE_PROTOBUF protobuf)
#find_program(HAVE_PROTOC "protoc")
#if (NOT HAVE_PROTOBUF OR NOT HAVE_PROTOC)
#  message(FATAL_ERROR "Required protocol buffer headers and compiler not found.")
#else()
#  set(PROTOBUF_LIBS protobuf)
#endif()

# -------- MPI --------

find_package(MPI REQUIRED)
message(STATUS "MPI: ${MPI_CXX_FOUND} ${MPIEXEC}")
#include_directories(SYSTEM ${MPI_CXX_INCLUDE_PATH})
# ${MPI_CXX_LINK_FLAGS}     Linking flags for MPI programs
# ${MPI_CXX_LIBRARIES}      All libraries to link MPI programs against
message(STATUS "mpi-include-path ${MPI_CXX_INCLUDE_PATH}")
message(STATUS "mpi-compile-flags ${MPI_CXX_COMPILE_FLAGS}")
message(STATUS "mpi-link-flags ${MPI_CXX_LINK_FLAGS}")
message(STATUS "mpi-libs ${MPI_CXX_LIBRARIES}")

# -------- compress (LZO) --------
#check_include_file("/usr/include/lzo/lzo1x.h" HAVE_LZO1X_H)
#if (NOT HAVE_LZO1X_H)
#  message(FATAL_ERROR "Required lzo header file not found.")
#endif()

if ("$ENV{OPROFILE}")
  message(STATUS "Enabling OProfile")
endif()

if( CMAKE_SIZEOF_VOID_P EQUAL 8 )
	set( ARCH_BITS 64 )
else()
	set( ARCH_BITS 32 )
endif()

# -------- thread --------

find_package (Threads)
message(STATUS "Thread lib: ${CMAKE_THREAD_LIBS_INIT}")
# use ${CMAKE_THREAD_LISBS_INIT} for targets need thread support.

# -------- main --------

message(STATUS "We are on a ${CMAKE_SYSTEM_NAME} system")
message(STATUS "The host processor is ${CMAKE_HOST_SYSTEM_PROCESSOR}")
message(STATUS "Source dir: ${PROJECT_SOURCE_DIR}")
message(STATUS "Binary dir: ${PROJECT_BINARY_DIR}")

#set(CMAKE_INCLUDE_CURRENT_DIR ON)

#include_directories( ${CMAKE_CURRENT_SOURCE_DIR}
#                     ${PROTOBUF_INCLUDE_DIRS}
#                     ${MPI_CXX_INCLUDE_PATH}
 #                  )

include_directories(SYSTEM)

#set(CMAKE_INCLUDE_CURRENT_DIR ON)

message("current dir: ${CMAKE_CURRENT_SOURCE_DIR}")

set(CXX_DEFAULT "${CMAKE_CXX_FLAGS}")
set(CXX_DEFINE "-DELPP_THREAD_SAFE -DELPP_STL_LOGGING")
#set(CXX_WARN "-Wall -Wno-unused-function -Wno-unused-variable -Wno-unused-result")
set(CMAKE_CXX_FLAGS "${CXX_DEFAULT} ${CXX_WARN} ${CXX_DEFINE}")

get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
message(STATUS ${INCLUDE_DIRECTORIES})
foreach(dir ${dirs})
  message(STATUS "dir='${dir}'")
endforeach()

add_subdirectory(src)

#add_subdirectory(test)

