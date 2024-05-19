## For compiling:
#### Pthread
```bash
gcc main.c socket_server.c cJSON.c -o a.out -Wall -lpthread -g
```
##### CUDA
```bash
nvcc main_cuda.cu cJSON.c win_socket_server.c -o cuda && timeout 5 && cuda.exe 
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