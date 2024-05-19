## For compiling:
#### Pthread
```bash
gcc main.c socket_server.c cJSON.c -o a.out -Wall -lpthread -g
```
##### CUDA
```bash
nvcc -G main_cuda.cu -o cuda
```

## For running:
First change the configuration in the `config.json` file.

For running the "backend"(**Pthread**):
```bash
./a.out
```

For running the "backend"(**CUDA**):
```bash
cuda.exe
```

For running the "frontend":
```bash
python3 displayer.py
```


## Lil diagram

![diagram](diagram.png)