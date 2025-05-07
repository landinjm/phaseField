##
#  CMake script for the PRISMS-PF applications
#  Adapted from the ASPECT CMake file
##

cmake_minimum_required(VERSION 3.13.4)

# Find the PRISMS-PF package
find_package(PRISMS-PF REQUIRED)

# Grab the macros
include(setup_application)
include(macros/macro_prisms_pf_autopilot)

# Create project
project(myapp CXX)

# Run autopilot
prisms_pf_autopilot(${PRISMS_PF_CORE_DIR})
