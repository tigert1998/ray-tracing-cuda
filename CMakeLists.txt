project(ray-tracing-cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)

add_subdirectory(third_party)

set(rt_root_dir "ray-tracing-cuda")
file(GLOB_RECURSE rt_srcs "${rt_root_dir}/*.cu")
add_library(rt STATIC "${rt_srcs}")
target_link_libraries(rt glm)
target_include_directories(rt PUBLIC "${rt_root_dir}")
set_target_properties(rt PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_features(rt PUBLIC cxx_std_14)

add_executable(hello "scenes/main.cu")
target_link_libraries(hello rt)
set_target_properties(hello PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_features(hello PUBLIC cxx_std_14)