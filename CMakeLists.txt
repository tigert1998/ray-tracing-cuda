project(ray-tracing-cuda LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)

add_subdirectory(third_party)

set(rt_root_dir "ray-tracing-cuda")
file(GLOB_RECURSE rt_srcs "${rt_root_dir}/*.cc")
add_library(rt STATIC "${rt_srcs}")
target_link_libraries(rt glm)