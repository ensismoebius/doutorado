##
## dataLoaders_10_1117_windowing — sliding-window datasets for EEG + audio multimodal pipelines.
##
## Provides:
##   EEGWindowDataset   - Option 2 EEG pipeline.
##   AudioWindowDataset - Option 2 Audio pipeline.
##   FusedWindowDataset - Option 3 fused (EEG + Audio) pipeline.
##
## This module depends on the base 10.1117 loaders for schema/session utilities.
##

add_library(dataLoaders_10_1117_windowing
    STATIC
        EEGWindowDataset.cpp
        AudioWindowDataset.cpp
        FusedWindowDataset.cpp
)

target_include_directories(dataLoaders_10_1117_windowing
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(dataLoaders_10_1117_windowing
    PUBLIC
        tensor
        matioCpp
        dataLoaders
        dataLoaders_10_1117
    PRIVATE
        Eigen3::Eigen
)
