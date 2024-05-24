## For compiling and running:
#### Pthread
```bash
gcc main.c socket_server.c cJSON.c -o a.out -Wall -lpthread -g
```
##### CUDA
```bash
nvcc main_cuda.cu cJSON.c win_socket_server.c -o cuda && cuda.exe
```
#### OpenMP
```bash
gcc -fopenmp -o openmp main_openmp.c && openmp.exe
```

## For running the "frontend":
```bash
python3 displayer.py
```


## Lil diagram

![diagram](diagram.png)