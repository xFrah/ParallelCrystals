## For compiling and running:

##### CUDA
```bash
nvcc main_cuda.cu common.c libs/cJSON.c libs/win_socket_server.c -o main_cuda && main_cuda.exe
```
##### OpenMP
```bash
gcc -fopenmp main_openmp.cpp common.c libs/cJSON.c libs/win_socket_server.c -o main_openmp -lstdc++ -lws2_32
```

## For running the "frontend":
```bash
python3 displayer.py
```
