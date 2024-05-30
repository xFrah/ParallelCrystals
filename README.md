## For compiling and running:

##### Single thread
```bash
gcc main.cpp common.c libs/cJSON.c libs/win_socket_server.c -o main -lstdc++ -lws2_32 -static && main.exe
```

##### CUDA
```bash
nvcc main_cuda.cu common.c libs/cJSON.c libs/win_socket_server.c -o main_cuda && main_cuda.exe
```
##### OpenMP
```bash
gcc -fopenmp main_openmp.cpp common.c libs/cJSON.c libs/win_socket_server.c -o main_openmp -lstdc++ -lws2_32 -static && main_openmp.exe
```

## For running the "frontend":
```bash
python3 displayer.py
```
