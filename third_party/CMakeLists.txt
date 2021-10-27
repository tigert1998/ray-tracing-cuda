project(third-party)

# glm
set(glm_dir "glm")
add_library(glm INTERFACE)
target_include_directories(glm INTERFACE "${glm_dir}")

# glog
add_subdirectory(glog)