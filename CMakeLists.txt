project(ray-tracing-cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)

add_subdirectory(third_party)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --extended-lambda")

set(ray_tracing_root_dir "ray-tracing-cuda")
file(GLOB_RECURSE ray_tracing_srcs "${ray_tracing_root_dir}/*.cu")
add_library(ray_tracing STATIC "${ray_tracing_srcs}")
target_link_libraries(ray_tracing glm glog::glog stb mpi librt.so)
target_include_directories(ray_tracing PUBLIC "${ray_tracing_root_dir}")
set_target_properties(ray_tracing PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_features(ray_tracing PUBLIC cxx_std_14)

add_executable(spheres "scenes/spheres.cu")
target_link_libraries(spheres ray_tracing)
set_target_properties(spheres PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_features(spheres PUBLIC cxx_std_14)

add_executable(cornell_box "scenes/cornell_box.cu")
target_link_libraries(cornell_box ray_tracing)
set_target_properties(cornell_box PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_features(cornell_box PUBLIC cxx_std_14)

add_executable(bunny "scenes/bunny.cu")
target_link_libraries(bunny ray_tracing assimp)
set_target_properties(bunny PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_features(bunny PUBLIC cxx_std_14)
