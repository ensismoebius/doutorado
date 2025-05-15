# Package finder
find_package(PkgConfig REQUIRED)
# Find Eigen

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Find OpenMP
find_package(OpenMP REQUIRED)