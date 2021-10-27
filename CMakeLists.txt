project(ray-tracing-cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)

add_subdirectory(third_party)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

set(ray_tracing_root_dir "ray-tracing-cuda")
file(GLOB_RECURSE ray_tracing_srcs "${ray_tracing_root_dir}/*.cu")
add_library(ray_tracing STATIC "${ray_tracing_srcs}")
target_link_libraries(ray_tracing glm glog::glog librt.so)
target_include_directories(ray_tracing PUBLIC "${ray_tracing_root_dir}")
set_target_properties(ray_tracing PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_features(ray_tracing PUBLIC cxx_std_14)

add_executable(hello "scenes/main.cu")
target_link_libraries(hello ray_tracing)
set_target_properties(hello PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_features(hello PUBLIC cxx_std_14)