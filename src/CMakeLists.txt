# This file collects the source files for PRISMS-PF
set(PRISMS_PF_SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/dummy.cc)
foreach(TARGET IN ITEMS
  core
  solvers
  user_inputs
  utilities
)
  add_subdirectory(${TARGET})
endforeach()
set(PRISMS_PF_SOURCE_FILES ${PRISMS_PF_SOURCE_FILES} PARENT_SCOPE)