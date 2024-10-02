##
#  CMake script for the phaseField applications:
##


# Set the name of the project and target:
set(TARGET "main")

# Declare all source files the target consists of:
set(TARGET_SRC
  ${TARGET}.cc
)

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)

FIND_PACKAGE(deal.II 9.2.0 QUIET
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
)

DEAL_II_INITIALIZE_CACHED_VARIABLES()

project(${TARGET})

include_directories(${CMAKE_SOURCE_DIR}/../../include)
include_directories(${CMAKE_SOURCE_DIR}/../../src)
include_directories(${CMAKE_SOURCE_DIR})

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
	set(DEAL_II_CXX_FLAGS_DEBUG "${DEAL_II_CXX_FLAGS_DEBUG} -Wno-maybe-uninitialized -Wno-unused-parameter -Wno-extra")
	set(DEAL_II_CXX_FLAGS_RELEASE "${DEAL_II_CXX_FLAGS_RELEASE} -Wno-maybe-uninitialized -Wno-unused-parameter -Wno-extra")
endif()


DEAL_II_INVOKE_AUTOPILOT()
