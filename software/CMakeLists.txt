cmake_minimum_required(VERSION 3.9)

if (APPLE)
	set (CMAKE_OSX_DEPLOYMENT_TARGET "10.15" CACHE STRING
	"Minimum OS X deployment version" FORCE)

	# Fix behavior of CMAKE_CXX_STANDARD when targeting macOS.
	if (POLICY CMP0025)
		cmake_policy(SET CMP0025 NEW)
	endif ()
endif()

project(Doutorado
	VERSION 1.0
	DESCRIPTION "Many things!!!"
	LANGUAGES CXX)

# C++ version
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

#set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")

message(STATUS "***********************COMPILING FOR:***********************")
message(STATUS ${CMAKE_SYSTEM})
message(STATUS "************************************************************")

###############################################
###############################################
### Executable and libraries configuration  ###
###############################################
###############################################

# For visualizations
find_package(SFML 2 REQUIRED COMPONENTS graphics system window)
# For matrices operations
find_package(Armadillo REQUIRED)
# For paralelism
find_package(Threads REQUIRED)
find_package(OpenMP)

set (SRCS
src/main.cpp
src/lib/gaussian/gaussian.cpp
src/experiments/experiment00.cpp
src/experiments/experiment01.cpp
)

# Include dirs
include_directories(${ARMADILLO_INCLUDE_DIRS})

if(OpenMP_CXX_FOUND)
    include_directories(${OpenMP_CXX_INCLUDE_DIRS})
endif()

# Adding sfmlplot library project
add_subdirectory(src/lib/sfmlplot)

# Executable
add_executable(${CMAKE_PROJECT_NAME} ${SRCS})

# User libraries
target_include_directories(${CMAKE_PROJECT_NAME}
	PUBLIC src/lib/gaussian
)

# Link the executable to the sfmlplot library. Since the sfmlplot library has
# public include directories we will use those link directories when building
target_link_libraries (${CMAKE_PROJECT_NAME} LINK_PUBLIC sfmlplot)

# Libraries to link
set(SFML_LIBS sfml-system sfml-window sfml-graphics)
target_link_libraries(${CMAKE_PROJECT_NAME} LINK_PUBLIC ${SFML_LIBS})

if(OpenMP_CXX_FOUND)
    target_link_libraries(${CMAKE_PROJECT_NAME} LINK_PUBLIC OpenMP::OpenMP_CXX)
endif()

target_link_libraries(${CMAKE_PROJECT_NAME} LINK_PUBLIC ${ARMADILLO_LIBRARIES})

# Copies the used font to folder
file(
    COPY ${CMAKE_CURRENT_SOURCE_DIR}/res/fonts/Arial.ttf
    DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/res/fonts
)

########################################
########################################
### Google Unit Testing suite setup  ###
########################################
########################################

if (NOT APPLE)
	# Locate GTest
	find_package(GTest REQUIRED)
	include_directories(${GTEST_INCLUDE_DIRS})

	# Load paraconsistent tests
	#add_executable( ${PROJECT_NAME}Tests
	#)

	# Link paraconsistentTests with what we want to test and the GTest and pthread library
	# target_link_libraries(${PROJECT_NAME} PUBLIC ${GTEST_LIBRARIES} pthread)
endif()
