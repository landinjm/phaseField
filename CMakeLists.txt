##
#  CMake script for the PRISMS-PF applications:
##

CMAKE_MINIMUM_REQUIRED(VERSION 3.1.0)

# Find deal.II installation
find_package(deal.II 9.2.0 QUIET REQUIRED
	HINTS ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR})
DEAL_II_INITIALIZE_CACHED_VARIABLES()

# Set up the debug, release, and run targets
ADD_CUSTOM_TARGET(debug
  COMMAND +env ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ${CMAKE_SOURCE_DIR}
  COMMAND +env ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
  COMMENT "Switch CMAKE_BUILD_TYPE to Debug"
  )

ADD_CUSTOM_TARGET(release
  COMMAND +env ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release ${CMAKE_SOURCE_DIR}
  COMMAND +env ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
  COMMENT "Switch CMAKE_BUILD_TYPE to Release"
  )

ADD_CUSTOM_TARGET(run COMMAND main
  COMMENT "Run with ${CMAKE_BUILD_TYPE} configuration"
)


# Set up the paths to the library files
include_directories(include)

file(GLOB_RECURSE headers include/*.h)
file(GLOB_RECURSE sources src/*.cc)

# Set source files
set(SOURCE_FILES
  ${headers}
  ${sources}
)

# Add main project
PROJECT(prisms_pf)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(${PROJECT_NAME} ${SOURCE_FILES})
DEAL_II_SETUP_TARGET(${PROJECT_NAME})

PROJECT(prisms_pf_debug)
SET(CMAKE_BUILD_TYPE Debug)

add_library(${PROJECT_NAME} ${SOURCE_FILES})
DEAL_II_SETUP_TARGET(${PROJECT_NAME})
