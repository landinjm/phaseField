##
# CMake for PRISMS-PF unit tests
##

message(STATUS "Setting up PRISMS-PF unit tests...")

# Add each test subdirectory
# add_subdirectory(core) # TODO: Re-enable when core tests are added
add_subdirectory(field_input)
add_subdirectory(user_inputs)
add_subdirectory(utilities)

message(STATUS "PRISMS-PF unit tests setup complete.")
