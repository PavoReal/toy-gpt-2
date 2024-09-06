# toy-gpt-2
A toy implementation of GPT-2 written in C++ for my own learning purposes. SDL/IMGUI/Opengl are used to visualize various aspects of the model and math behind the model. Initial implementation reference is [picoGPT](https://github.com/jaymody/picoGPT).

## TODO
- [ ] Implement very simple versions of math functions
    - [X] GELU
    - [ ] Softmax
    - [ ] Layer Normalization
    - [ ] Linear
    - [ ] FFN
    - [ ] Attention
    - [ ] MHA
- [ ] Implement visualization for math functions
    - [X] GELU
    - [ ] Softmax
    - [ ] Layer Normalization
    - [ ] Linear
    - [ ] FFN
    - [ ] Attention
    - [ ] MHA
- [ ] Implement GPT-2 model

## Dependencies
All dependencies are git submodules within `deps` directory.

To initialize submodules, run:
```bash
git submodule update --init --recursive
```
The dependencies are:
- [SDL2](https://github.com/libsdl-org/SDL/tree/SDL2)
- [IMGUI](https://github.com/ocornut/imgui)

## Building
CMake is used as the build system and *should* handle everything. To build, initalize submodules and run the following:

```bash
mkdir build
cd build
cmake ..
cmake --build .
```
