## For compiling and running:

##### CUDA
```bash
nvcc main_cuda.cu common.c libs/cJSON.c libs/win_socket_server.c -o main_cuda && main_cuda.exe
```

## For running the "frontend":
```bash
python3 displayer.py
```
