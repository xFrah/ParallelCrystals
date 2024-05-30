## Files explanation:
#### Scripts:
- `main.cpp` - Main file for the single-threaded version.
- `main_cuda.cu` - Main file for the CUDA version.
- `main_openmp.cpp` - Main file for the OpenMP version.
- `displayer.py` - Python script to display the results in a GUI.
- `algorithm_showcase.py` - Python script for the showcase of the algorithm.
#### Libraries:
- `common.c` - Common functions for all versions.
- `libs/cJSON.c` - cJSON library for JSON parsing.
- `libs/win_socket_server.c` - Windows socket server for communication with `displayer.py`.
#### Others:
- `config.json` - Configuration file for the algorithm.
- `trash` - Folder for old files and tests.

---

## For compiling and running:
First of all, modify `config.json` to your needs.

If "show_visually" is set to true, you will need to run `displayer.py` while running the main file.

#### Single thread
```bash
gcc main.cpp common.c libs/cJSON.c libs/win_socket_server.c -o main -lstdc++ -lws2_32 -static && main.exe
```

#### CUDA
```bash
nvcc main_cuda.cu common.c libs/cJSON.c libs/win_socket_server.c -o main_cuda && main_cuda.exe
```
#### OpenMP
```bash
gcc -fopenmp main_openmp.cpp common.c libs/cJSON.c libs/win_socket_server.c -o main_openmp -lstdc++ -lws2_32 -static && main_openmp.exe
```

## For displaying the results:
Set "show_visually" to true in `config.json` and run the following command:
```bash
python3 displayer.py
```
Then run the main file, if you haven't already.