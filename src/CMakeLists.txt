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
message(STATUS ${CMAKE_TOOLCHAIN_FILE})
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
)

# Include dirs
include_directories(${ARMADILLO_INCLUDE_DIRS})

if(OpenMP_CXX_FOUND)
    include_directories(${OpenMP_CXX_INCLUDE_DIRS})
endif()

# Executable
add_executable(${CMAKE_PROJECT_NAME} ${SRCS})

# User libraries
target_include_directories(${CMAKE_PROJECT_NAME}
	PUBLIC src/lib/gaussian
)

# Libraries to link
set(SFML_LIBS sfml-system sfml-window sfml-graphics)
target_link_libraries(${PROJECT_NAME} ${SFML_LIBS})

if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX)
endif()

target_link_libraries(${PROJECT_NAME} ${ARMADILLO_LIBRARIES})

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
	target_link_libraries(${PROJECT_NAME}Tests PUBLIC ${GTEST_LIBRARIES} pthread)
endif()
