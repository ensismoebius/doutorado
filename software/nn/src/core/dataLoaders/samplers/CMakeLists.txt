add_library(dataLoaders_samplers
    STATIC
        SequentialSampler.cpp
        RandomSampler.cpp
        WeightedRandomSampler.cpp
        DistributedSampler.cpp
)

target_include_directories(dataLoaders_samplers
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
)

add_subdirectory(tests)
