add_executable(meeting01_comparative_gtest
    ComparativeExperiment_gtest.cpp
)

target_include_directories(meeting01_comparative_gtest
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(meeting01_comparative_gtest
    PRIVATE
        google_test
        tensor
        layers
        data_loaders
        util
        xtensor xtensor-blas
        statistics
        matioCpp
        dataLoaders_10_1117_windowing
        dataLoaders_10_1117
        cnpy-static
)

include(GoogleTest)
gtest_discover_tests(meeting01_comparative_gtest)
configure_xtensor_parallel_target(meeting01_comparative_gtest)

# DAT writer tests
add_executable(dat_writer_gtest
    dat_writer_gtest.cpp
)

target_include_directories(dat_writer_gtest
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(dat_writer_gtest
    PRIVATE
        google_test
        meeting01_lib
        tensor
        layers
        xtensor xtensor-blas
        cnpy-static
)

gtest_discover_tests(dat_writer_gtest)
configure_xtensor_parallel_target(dat_writer_gtest)

# Profile audit: every shipping JSON profile must parse, validate, and
# populate every config field the harness consumes (catches silent drift).
add_executable(profile_audit_gtest
    profile_audit_gtest.cpp
)

target_include_directories(profile_audit_gtest
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(profile_audit_gtest
    PRIVATE
        google_test
        meeting01_lib
        xtensor xtensor-blas
        cnpy-static
)

gtest_discover_tests(profile_audit_gtest)
configure_xtensor_parallel_target(profile_audit_gtest)

# EDF-format EEG loader on hand-built fixtures (no corpus needed, always registered).
add_executable(loaders_gtest
    loaders_gtest.cpp
)

target_include_directories(loaders_gtest
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_SOURCE_DIR}/src/experiments/meeting01/lib/src
)

target_link_libraries(loaders_gtest
    PRIVATE
        google_test
        meeting01_lib
        data_loaders_fsdd
        xtensor xtensor-blas
        cnpy-static
)

gtest_discover_tests(loaders_gtest)
configure_xtensor_parallel_target(loaders_gtest)

# The real corpora through the real loaders (FSDD, AudioMNIST, MIT-BIH, eegmmidb, Siena).
# They are large and not in the repository, so ctest registers them only on request:
#   cmake --preset=max-performance -DNN_TEST_DB_ROOT=<directory holding the corpora>
# Each suite is registered when its corpus directory exists under that root, and this file says
# which ones it left out. A registered test never skips: a missing corpus fails it (see
# loaders_corpus_gtest.cpp). The target itself is always built, so it cannot rot.
add_executable(loaders_corpus_gtest
    loaders_corpus_gtest.cpp
)

target_include_directories(loaders_corpus_gtest
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_SOURCE_DIR}/src/experiments/meeting01/lib/src
)

target_link_libraries(loaders_corpus_gtest
    PRIVATE
        google_test
        meeting01_lib
        data_loaders_fsdd
        xtensor xtensor-blas
        cnpy-static
)

configure_xtensor_parallel_target(loaders_corpus_gtest)

set(NN_TEST_DB_ROOT "" CACHE PATH
    "Directory holding the real corpora for loaders_corpus_gtest (empty: those tests are not registered)")

if(NN_TEST_DB_ROOT STREQUAL "")
    message(STATUS "loaders_corpus_gtest: NN_TEST_DB_ROOT not set, its real-corpus tests are not "
                   "registered with ctest (configure with -DNN_TEST_DB_ROOT=<dir> to run them)")
else()
    if(NOT IS_DIRECTORY "${NN_TEST_DB_ROOT}")
        message(FATAL_ERROR "NN_TEST_DB_ROOT='${NN_TEST_DB_ROOT}' is not a directory. "
                            "Point it at the directory that holds the corpora, or leave it empty.")
    endif()
    # suite name : corpus directory name
    set(_nn_corpora
        CorpusFsdd:fsdDataset
        CorpusAudioMnist:audioMNIST_8k
        CorpusMitBih:mitbih
        CorpusEegmmidb:eegmmidb
        CorpusSiena:siena)
    set(_nn_corpus_filter "")
    foreach(_pair IN LISTS _nn_corpora)
        string(REPLACE ":" ";" _parts "${_pair}")
        list(GET _parts 0 _suite)
        list(GET _parts 1 _dir)
        if(IS_DIRECTORY "${NN_TEST_DB_ROOT}/${_dir}")
            list(APPEND _nn_corpus_filter "${_suite}.*")
            message(STATUS "loaders_corpus_gtest: ${_suite} registered (${NN_TEST_DB_ROOT}/${_dir})")
        else()
            message(STATUS "loaders_corpus_gtest: ${_suite} NOT registered, no '${_dir}' under "
                           "NN_TEST_DB_ROOT")
        endif()
    endforeach()
    if(_nn_corpus_filter)
        list(JOIN _nn_corpus_filter ":" _nn_corpus_filter)
        gtest_discover_tests(loaders_corpus_gtest
            TEST_FILTER "${_nn_corpus_filter}"
            PROPERTIES ENVIRONMENT "NN_TEST_DB_ROOT=${NN_TEST_DB_ROOT}")
    endif()
endif()

# Structured JSONL event sink for the live monitor (schema helpers + round-trip).
add_executable(event_sink_gtest
    event_sink_gtest.cpp
)

target_include_directories(event_sink_gtest
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_SOURCE_DIR}/src/experiments/meeting01/lib/src
)

target_link_libraries(event_sink_gtest
    PRIVATE
        google_test
        meeting01_lib
        xtensor xtensor-blas
        cnpy-static
)

gtest_discover_tests(event_sink_gtest)
configure_xtensor_parallel_target(event_sink_gtest)

# Parameter-count audit: documented P for LSTM-/GRU-/Transformer-AE at the article config.
add_executable(parameter_count_gtest
    parameter_count_gtest.cpp
)

target_include_directories(parameter_count_gtest
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(parameter_count_gtest
    PRIVATE
        google_test
        meeting01_lib
        gru_models
        transformer_models
        tensor
        layers
        xtensor xtensor-blas
        cnpy-static
)

gtest_discover_tests(parameter_count_gtest)
configure_xtensor_parallel_target(parameter_count_gtest)

# Encoding audit: latency is canonical single-spike TTFS; poisson normalizes over
# the full [min, max] range (not max-only), so sub-mean/negative post-z-score
# samples keep a real firing probability.
add_executable(meeting01_encoding_gtest
    meeting01_encoding_gtest.cpp
)

target_include_directories(meeting01_encoding_gtest
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(meeting01_encoding_gtest
    PRIVATE
        google_test
        meeting01_lib
        tensor
        xtensor xtensor-blas
        cnpy-static
)

gtest_discover_tests(meeting01_encoding_gtest)
configure_xtensor_parallel_target(meeting01_encoding_gtest)

# Split audit: nested LOSO fold assignment is speaker/recording/window disjoint.
add_executable(meeting01_split_audit_gtest
    split_audit_gtest.cpp
)

target_include_directories(meeting01_split_audit_gtest
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(meeting01_split_audit_gtest
    PRIVATE
        google_test
        meeting01_lib
        tensor
        layers
        data_loaders
        xtensor xtensor-blas
        cnpy-static
)

gtest_discover_tests(meeting01_split_audit_gtest)
configure_xtensor_parallel_target(meeting01_split_audit_gtest)

# NSGA-II SNN-AE architecture search: genome repair/crossover/mutate legality,
# to_ae_config free-form rendering (+ the fixed-profile regression anchor), checkpoint
# round-trip, and the shared ga::Nsga2Core contract via Meeting01GaIndividual.
add_executable(meeting01_ga_gtest
    meeting01_ga_gtest.cpp
)

target_include_directories(meeting01_ga_gtest
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(meeting01_ga_gtest
    PRIVATE
        google_test
        meeting01_lib
        tensor
        layers
        xtensor xtensor-blas
        cnpy-static
)

gtest_discover_tests(meeting01_ga_gtest)
configure_xtensor_parallel_target(meeting01_ga_gtest)

# Shared LSTM-AE / GRU-AE architecture search: RecurrentGenome repair/crossover/mutate
# legality, to_lstm_cfg/to_gru_cfg regression anchors (seq_len formula, fixed
# latent_dim), checkpoint round-trip for both LstmGaIndividual and GruGaIndividual.
add_executable(meeting01_recurrent_ga_gtest
    meeting01_recurrent_ga_gtest.cpp
)

target_include_directories(meeting01_recurrent_ga_gtest
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(meeting01_recurrent_ga_gtest
    PRIVATE
        google_test
        meeting01_lib
        tensor
        layers
        xtensor xtensor-blas
        cnpy-static
)

gtest_discover_tests(meeting01_recurrent_ga_gtest)
configure_xtensor_parallel_target(meeting01_recurrent_ga_gtest)

# Transformer-AE architecture search: TransformerGenome repair/crossover/mutate
# legality (including the d_model % n_heads == 0 invariant unique to this family),
# to_transformer_cfg regression anchor (seq_len formula, fixed latent_dim), checkpoint
# round-trip for TransformerGaIndividual.
add_executable(meeting01_transformer_ga_gtest
    meeting01_transformer_ga_gtest.cpp
)

target_include_directories(meeting01_transformer_ga_gtest
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(meeting01_transformer_ga_gtest
    PRIVATE
        google_test
        meeting01_lib
        tensor
        layers
        xtensor xtensor-blas
        cnpy-static
)

gtest_discover_tests(meeting01_transformer_ga_gtest)
configure_xtensor_parallel_target(meeting01_transformer_ga_gtest)

# LSTM microbench (ops breakdown)
add_executable(lstm_ops_microbench
    lstm_ops_microbench.cpp
)

target_include_directories(lstm_ops_microbench
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(lstm_ops_microbench
    PRIVATE
        tensor
        layers
        xtensor xtensor-blas
)

configure_xtensor_parallel_target(lstm_ops_microbench)
