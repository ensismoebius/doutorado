##
## Trainer unit tests (GoogleTest)
##
## Builds:
## - `trainer_gtest`: tests for TrainerConfig and EpochResult
##
## Note: Trainer template has pre-existing bugs in template body that need fixing
## before full Trainer testing is possible. Only Config and EpochResult are tested.
##

add_executable(trainer_gtest
    trainer_gtest.cpp
)

target_link_libraries(trainer_gtest
    PRIVATE
        google_test
        tensor
        layers
        optimizers
)

target_include_directories(
    trainer_gtest
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src/core/training
)

gtest_discover_tests(trainer_gtest)

##
## trainer_genericity_gtest: tests for Trainer<ModelType, LossType>, callbacks, EarlyStopping
##
add_executable(trainer_genericity_gtest
    trainer_genericity_gtest.cpp
)

target_link_libraries(trainer_genericity_gtest
    PRIVATE
        google_test
        tensor
        layers
        optimizers
        progress
)

target_include_directories(
    trainer_genericity_gtest
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src/core/training
)

gtest_discover_tests(trainer_genericity_gtest)