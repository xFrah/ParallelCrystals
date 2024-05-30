#include "win_socket_server.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <winsock2.h>
#include <ws2tcpip.h>

#pragma comment(lib, "ws2_32.lib")

void init_winsock() {
    WSADATA wsaData;
    int iResult = WSAStartup(MAKEWORD(2, 2), &wsaData);
    if (iResult != 0) {
        printf("WSAStartup failed: %d\n", iResult);
        exit(EXIT_FAILURE);
    }
}

void cleanup_winsock() {
    WSACleanup();
}

int socket_server_start(int port) {
    init_winsock();

    // Creating socket file descriptor
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd == INVALID_SOCKET) {
        perror("socket failed");
        cleanup_winsock();
        exit(EXIT_FAILURE);
    }

    struct sockaddr_in address;
    int new_socket;
    int addrlen = sizeof(address);
    int opt = 1;

    // Forcefully attaching socket to the port
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, (char*)&opt, sizeof(opt)) == SOCKET_ERROR) {
        perror("setsockopt");
        closesocket(server_fd);
        cleanup_winsock();
        exit(EXIT_FAILURE);
    }

    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port);

    // Binding socket to the port
    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) == SOCKET_ERROR) {
        perror("bind failed");
        closesocket(server_fd);
        cleanup_winsock();
        exit(EXIT_FAILURE);
    }

    if (listen(server_fd, 3) == SOCKET_ERROR) {
        perror("listen");
        closesocket(server_fd);
        cleanup_winsock();
        exit(EXIT_FAILURE);
    }

    printf("Listening on socket\n");

    while (1) {
        new_socket = accept(server_fd, (struct sockaddr *)&address, &addrlen);
        if (new_socket != INVALID_SOCKET) {
            break;
        }
    }

    printf("Connection accepted\n");
    return new_socket;
}

void socket_server_send(int socket, void *particles, int length) {
    int buf_lenght = 4 + 4 + length + 4;
    char *buffer = (char *)malloc(buf_lenght);
    memcpy(buffer, "PART", 4);
    memcpy(buffer + 4, &length, 4);
    memcpy(buffer + 8, particles, length);
    memcpy(buffer + 8 + length, "ENDP", 4);
    int i = send(socket, buffer, buf_lenght, 0);
    if (i < 0) {
        perror("send");
        closesocket(socket);
        cleanup_winsock();
        exit(EXIT_FAILURE);
    }
}
