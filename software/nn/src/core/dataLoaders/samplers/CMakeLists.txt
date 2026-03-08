add_library(dataLoaders_samplers
    STATIC
        RandomSampler.cpp
        SequentialSampler.cpp
        DistributedSampler.cpp
        WeightedRandomSampler.cpp
)

target_include_directories(dataLoaders_samplers
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
)

add_subdirectory(tests)
