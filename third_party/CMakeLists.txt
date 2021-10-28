project(third-party)

# glm
set(glm_dir "glm")
add_library(glm INTERFACE)
target_include_directories(glm INTERFACE "${glm_dir}")

# glog
add_subdirectory(glog)

# assimp
find_package(assimp REQUIRED)
add_library(assimp INTERFACE)
target_include_directories(assimp INTERFACE "${ASSIMP_INCLUDE_DIRS}")
target_link_libraries(assimp INTERFACE "${ASSIMP_LIBRARY_DIRS}/${ASSIMP_LIBRARIES}")

# stb
set(stb_dir "stb")
add_library(stb INTERFACE)
target_include_directories(stb INTERFACE "${stb_dir}")

find_package(MPI REQUIRED)
add_library(mpi INTERFACE)
target_include_directories(mpi INTERFACE "${MPI_C_INCLUDE_PATH}")
target_link_libraries(mpi INTERFACE "${MPI_C_LIBRARIES}")
