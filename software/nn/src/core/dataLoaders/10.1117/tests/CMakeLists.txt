##
## 10.1117 dataset loader unit tests (GoogleTest)
##
## Builds:
## - `dataLoaders_10_1117_gtest`: tests for EEG/Audio MAT loaders following the
##   10.1117-style dataset schema.
##
## Notes:
## - Links against `dataLoaders_10_1117` and uses `MatTestUtilsLib` fixtures.
##
add_executable(dataLoaders_10_1117_gtest
	loaders/audio_loader_gtest.cpp
	loaders/audio_loader_article_spec_gtest.cpp
	fuzz/audio_loader_property_tests.cpp
	loaders/eeg_loader_gtest.cpp
	loaders/eeg_loader_edge_gtest.cpp
	loaders/eeg_loader_article_spec_gtest.cpp
	loaders/sqlite_session_gtest.cpp
	fuzz/eeg_loader_property_tests.cpp
	schema/dataset_schema_article_spec_gtest.cpp
	schema/subject_discovery_gtest.cpp
	codec/batch_target_formatter_gtest.cpp
	codec/input_mode_codec_gtest.cpp
	protocol/protocol101117_dataset_modes_gtest.cpp
	generators/mock_dataset_generator_gtest.cpp
	utils/MockImaginedSpeechDatasetGenerator.cpp
)

target_include_directories(dataLoaders_10_1117_gtest 
	PRIVATE 
		${CMAKE_SOURCE_DIR}/src 
		${CMAKE_CURRENT_SOURCE_DIR}
		${CMAKE_SOURCE_DIR}/src/core/dataLoaders/tests
)

target_link_libraries(dataLoaders_10_1117_gtest 
	PRIVATE 
		google_test 
		Eigen3::Eigen 
		${ZLIB_LIBRARIES} 
		matioCpp
        dataLoaders_10_1117 # Link to the new library
        MatTestUtilsLib
)

include(GoogleTest)
add_subdirectory(fuzz)
add_subdirectory(windowing)
configure_eigen_parallel_target(dataLoaders_10_1117_gtest)
gtest_discover_tests(dataLoaders_10_1117_gtest)
