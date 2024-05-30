nvcc main_cuda.cu common.c libs/cJSON.c libs/win_socket_server.c -o main_cuda
del main_cuda.exp main_cuda.lib
main_cuda.exe