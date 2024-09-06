cmake_minimum_required(VERSION 3.10)

# Set the project name and version
project(pico-gpt2-c VERSION 1.0)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(SDL2_STATIC ON)
add_subdirectory(deps/SDL)

# IMGUI sources
set(imgui_srcs
    deps/imgui/imgui.cpp
    deps/imgui/imgui_draw.cpp
    deps/imgui/imgui_tables.cpp
    deps/imgui/imgui_widgets.cpp
    deps/imgui/backends/imgui_impl_sdl2.cpp
    deps/imgui/backends/imgui_impl_opengl3.cpp
)

# Find OpenGL package
find_package(OpenGL REQUIRED)

add_executable(pico-gpt2-c src/main.cpp src/pico-gpt2.hpp src/utils.hpp ${imgui_srcs})

# Link OpenGL library
target_link_libraries(pico-gpt2-c PUBLIC SDL2::SDL2-static OpenGL::GL)
target_include_directories(pico-gpt2-c PUBLIC ${CMAKE_SOURCE_DIR}/deps/imgui ${CMAKE_SOURCE_DIR}/deps/imgui/backends ${CMAKE_SOURCE_DIR}/deps/SDL/include)
