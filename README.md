# AlphaZero implementation for Connect 4
This is an implementation of AlphaZero for Connect 4 on C++ using Visual Studio 2022. Training process was optimized for GTX 1650. Project also includes AlphaBeta and python bindings for convinient testing of models using python.
## Dependencies
### General
* ISO C++ 20
* libtorch 2.9.1+cu130
* NVIDIA CUDA Toolkit 13.1
* TSL Robin-Map 1.4.1
### Specific for python binding
* Python 3.11
* pybind11 3.0.1
### Libtorch libs
* torch_cuda.lib
* c10_cuda.lib
* torch_cpu.lib
* c10.lib
* caffe2_nvrtc.lib
* cudart.lib

You also need to copy libtorch/lib/*.dll files to your build folder, either manually or with after-build event
## Throughput analysis
<img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/37ea2cf1-0e38-47a6-a982-b68986ddf860" />

## Console parameters
 
| parameter | description | mandatory | default |
| :-------- | :---------- | :-------: | :-----: |
| --name | save folder name | ✅ | ❌|
| --dev | cpu or cuda | ❌ | cpu |
| --mcts | number of mcts batches | ❌ | 32 |
| --batch | number of mcts searched per batch | ❌ | 64 |
| --games | number of self-play games per iteration | ❌ | 2048 |
| --parallel | number of self-play games played simulteneously | ❌ | 256 |

## Results
<img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/6913c2e6-68d7-46eb-a7ef-d09487a8449c" />
<img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/5a92b403-0324-4f56-996e-6f776baa16b0" />

## To-do
* CMake insead of Visual Studio for proper dependency manager and cross-platform
* Pascal Pons game model and Zobrist hashing for more efficiency
* Add configuration file to set hyperparameters
