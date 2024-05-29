#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <windows.h>

#include <chrono>
#include <iostream>

// Define kernel launch parameters
#define numBlocks_ 68
#define threadsPerBlock_ 32

extern "C" {
#include "libs/cJSON.h"
#include "libs/win_socket_server.h"
#include "common.h"
}

struct ListHead {
    Particle* head;
}

curandState* states;
__device__ Configuration d_config;
__device__ int gridHeight;
__device__ int gridWidth;
__device__ int cellSize;
struct Configuration config;
int gridHeight_host;
int gridWidth_host;
int cellSize_host;
Particle* particles;

#define CHECK_CUDA_ERROR(call)                                                \
    {                                                                         \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA error in " << __FILE__ << " at line "          \
                      << __LINE__ << ": " << cudaGetErrorString(err) << "\n"; \
            exit(err);                                                        \
        }                                                                     \
    }

#define CHECK_LAST_ERROR()                                                    \
    {                                                                         \
        cudaError_t err = cudaGetLastError();                                 \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA error in " << __FILE__ << " at line "          \
                      << __LINE__ << ": " << cudaGetErrorString(err) << "\n"; \
            exit(err);                                                        \
        }                                                                     \
    }

__global__ void reset_linked_lists(ListHead*** grid) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < gridHeight && col < gridWidth) {
        ListHead* cell = grid[row][col];
        cell->head = NULL;
    }
}

__device__ void appendNode(ListHead* listHead, Particle* newHead) {
    Particle* oldHead;
    do {
        oldHead = listHead->head;          // Read the current head
        newHead->next_particle = oldHead;  // Set the new node's next to the current head
    } while (atomicCAS((unsigned long long int*)&(listHead->head), (unsigned long long int)oldHead, (unsigned long long int)newHead) != (unsigned long long int)oldHead);
}

__global__ void makeLinkedLists(ListHead*** grid, Particle* particles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;
    int start = (idx * d_config.NUM_PARTICLES) / num_threads;
    int end = ((idx + 1) * d_config.NUM_PARTICLES) / num_threads;
    int gridX, gridY;
    Particle* p;

    for (int i = start; i < end; i++) {
        p = &particles[i];
        gridX = p->x / cellSize;
        gridY = p->y / cellSize;

        if (gridX < gridWidth && gridY < gridHeight) {
            if (gridX >= gridWidth) {
                gridX = gridWidth - 1;
            }
            if (gridY >= gridHeight) {
                gridY = gridHeight - 1;
            }
            if (gridX < 0) {
                gridX = 0;
            }
            if (gridY < 0) {
                gridY = 0;
            }
            appendNode(grid[gridY][gridX], p);
        }
    }
}
__global__ void sort_single_cell_insertionSort(ListHead*** grid) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    const int MAX_DEPTH = d_config.NUM_PARTICLES + (d_config.NUM_PARTICLES * 0.2);  // Maximum depth to prevent infinite loop

    if (row < gridHeight && col < gridWidth) {
        ListHead* cell = grid[row][col];
        Particle* head = cell->head;

        if (head == NULL || head->next_particle == NULL) {
            return;  // If the cell is empty or has only one element, there's nothing to sort.
        }

        // Insertion sort on the linked list by x-coordinate
        Particle* sorted = NULL;
        Particle* current = head;
        int depth = 0;  // Depth counter for cycle detection

        while (current != NULL) {
            depth++;
            if (depth > MAX_DEPTH) {
                printf("Cycle detected or excessive depth in cell (%d, %d)\n", row, col);
                assert(0);
                break;
            }

            Particle* next = current->next_particle;
            if (sorted == NULL || current->x < sorted->x) {
                current->next_particle = sorted;
                sorted = current;
            } else {
                Particle* search = sorted;
                while (search->next_particle != NULL && search->next_particle->x < current->x) {
                    search = search->next_particle;
                    depth++;
                    if (depth > MAX_DEPTH) {
                        printf("Cycle detected or excessive depth in cell (%d, %d)\n", row, col);
                        assert(0);
                        break;
                    }
                }
                current->next_particle = search->next_particle;
                search->next_particle = current;
            }
            current = next;
        }

        // Update the cell's head to the new sorted list head
        cell->head = sorted;

        // Ensure the last particle's next is null
        Particle* tail = sorted;
        depth = 0;  // Reset depth counter for tail update
        while (tail != NULL && tail->next_particle != NULL) {
            tail = tail->next_particle;
            depth++;
            if (depth > MAX_DEPTH) {
                printf("Cycle detected or excessive depth while updating tail in cell (%d, %d)\n", row, col);
                assert(0);
                break;
            }
        }
        // No need to update cell->tail, just ensure the last particle's next is null
        if (tail != NULL) {
            tail->next_particle = NULL;
        }
    }
}

__device__ void check_collisions(ListHead* cell1, ListHead* cell2) {
    Particle* p1 = cell1->head;
    Particle* pj = cell2->head;
    int threshold = d_config.PARTICLE_RADIUS;

    while (p1 != NULL && pj != NULL) {
        while (pj != NULL && pj->x < p1->x - threshold) {
            pj = pj->next_particle;
        }

        Particle* pk = pj;
        while (pk != NULL && pk->x <= p1->x + threshold) {
            if (p1->walker != pk->walker && abs(p1->y - pk->y) <= threshold) {
                printf("Collision between particles %d and %d\n", p1->id, pk->id);
                p1->walker = 0;
                pk->walker = 0;
            }
            pk = pk->next_particle;
        }
        p1 = p1->next_particle;
    }
}

__global__ void check_for_collisions(ListHead*** grid) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < gridHeight && col < gridWidth) {
        ListHead* cell = grid[row][col];
        // check what kind of cell this is
        if (row == 0 && col == 0) {
            // top left corner: compare with cell at the right, cell below, cell at bottom right
            check_collisions(cell, grid[row][col + 1]);
            check_collisions(cell, grid[row + 1][col]);
            check_collisions(cell, grid[row + 1][col + 1]);

            check_collisions(cell, cell);
        } else if (row == 0 && col == gridWidth - 1) {
            // top right corner: compare with cell below and cell at bottom left
            check_collisions(cell, grid[row + 1][col]);
            check_collisions(cell, grid[row + 1][col - 1]);
            check_collisions(cell, cell);
        } else if (row == gridHeight - 1 && col == 0) {
            // bottom left corner: compare with cell at the right
            check_collisions(cell, grid[row][col + 1]);
            check_collisions(cell, cell);
        } else if (row == gridHeight - 1 && col == gridWidth - 1) {
            // bottom right corner: compare with no one
            check_collisions(cell, cell);
        } else if (row == 0) {
            // top edge: compare with cell at the right, cell at bottom left, cell below and cell at bottom right
            check_collisions(cell, grid[row][col + 1]);
            check_collisions(cell, grid[row + 1][col - 1]);
            check_collisions(cell, grid[row + 1][col]);
            check_collisions(cell, grid[row + 1][col + 1]);
            check_collisions(cell, cell);
        } else if (row == gridHeight - 1) {
            // bottom edge: compare with cell at the right
            check_collisions(cell, grid[row][col + 1]);
            check_collisions(cell, cell);
        } else if (col == 0) {
            // left edge: compare with cell at the right, cell below, cell at bottom right
            check_collisions(cell, grid[row][col + 1]);
            check_collisions(cell, grid[row + 1][col]);
            check_collisions(cell, grid[row + 1][col + 1]);
            check_collisions(cell, cell);
        } else if (col == gridWidth - 1) {
            // right edge: compare with cell below and cell at bottom left
            check_collisions(cell, grid[row + 1][col]);
            check_collisions(cell, grid[row + 1][col - 1]);
            check_collisions(cell, cell);
        } else {
            // middle: compare with cell at the right, cell below, cell at bottom right, cell at bottom left
            check_collisions(cell, grid[row][col + 1]);
            check_collisions(cell, grid[row + 1][col]);
            check_collisions(cell, grid[row + 1][col + 1]);
            check_collisions(cell, grid[row + 1][col - 1]);
            check_collisions(cell, cell);
        }
    }
}

// debug function to print all the linked lists in all the cells
__global__ void print_linked_lists(ListHead*** grid) {
    int counter = 0;

    for (int i = 0; i < gridHeight; i++) {
        for (int j = 0; j < gridWidth; j++) {
            ListHead* cell = grid[i][j];
            if (cell->head == NULL) {
                // printf("Cell (%d, %d) is empty\n", i, j);
                continue;
            }
            Particle* p = cell->head;
            printf("[%d] Cell (%d, %d): ", counter, i, j);
            while (p != NULL) {
                printf("%d (%d, %d) ", p->id, p->x, p->y);
                counter++;
                p = p->next_particle;
            }
            printf("\n");
        }
    }
}

__global__ void move_particles_kernel(Particle* particles, curandState* states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;
    int start = (idx * d_config.NUM_PARTICLES) / num_threads;
    int end = ((idx + 1) * d_config.NUM_PARTICLES) / num_threads;
    for (int i = start; i < end; i++) {  // move particles
        if (particles[i].walker) {
            particles[i].x = particles[i].x + 1 + (-2 * (curand(&states[idx]) % 2));
            particles[i].y = particles[i].y + 1 + (-2 * (curand(&states[idx]) % 2));
        }
        particles[i].next_particle = NULL;
    }
}

__global__ void init_particles_kernel(Particle* particles, curandState* states) {
    for (int i = 0; i < numBlocks_ * threadsPerBlock_; i++) {
        curand_init(d_config.SEED, i, 0, &states[i]);
    }
    for (int i = 0; i < d_config.NUM_PARTICLES; i++) {
        particles[i].id = i;
        particles[i].x = curand(&states[0]) % d_config.WIDTH;
        particles[i].y = curand(&states[0]) % d_config.HEIGHT;
        particles[i].walker = curand(&states[0]) % 2;
        particles[i].next_particle = NULL;
    }
}

__global__ void initializeGrid(ListHead*** grid) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < gridHeight && col < gridWidth) {
        grid[row][col] = (ListHead*)malloc(sizeof(ListHead));
        grid[row][col]->head = NULL;
    }
}

ListHead*** allocate_memory() {
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_config, &config, sizeof(Configuration)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(gridHeight, &gridHeight_host, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(gridWidth, &gridWidth_host, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(cellSize, &cellSize_host, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&particles, config.NUM_PARTICLES * sizeof(Particle)));
    CHECK_CUDA_ERROR(cudaMalloc(&states, numBlocks_ * threadsPerBlock_ * sizeof(curandState)));

    std::cout << "Grid size: " << gridHeight_host << "x" << gridWidth_host << std::endl;

    // Allocate memory for the grid on the host
    ListHead*** h_grid = (ListHead***)malloc(gridHeight_host * sizeof(ListHead**));
    for (int i = 0; i < gridHeight_host; ++i) {
        h_grid[i] = (ListHead**)malloc(gridWidth_host * sizeof(ListHead*));
    }

    std::cout << "Allocated memory for the grid on the host" << std::endl;

    // Allocate memory for the grid on the device
    ListHead*** d_grid;
    cudaMalloc(&d_grid, gridHeight_host * sizeof(ListHead**));
    for (int i = 0; i < gridHeight_host; ++i) {
        ListHead** d_row;
        cudaMalloc(&d_row, gridWidth_host * sizeof(ListHead*));
        cudaMemcpy(&d_grid[i], &d_row, sizeof(ListHead*), cudaMemcpyHostToDevice);
    }

    std::cout << "Allocated memory for the grid on the device" << std::endl;
    return d_grid;
}

int main() {
    config = get_configuration();
    cellSize_host = config.CELL_SIZE;
    gridHeight_host = config.HEIGHT / cellSize_host;
    gridWidth_host = config.WIDTH / cellSize_host;

    ListHead*** d_grid = allocate_memory();

    int socket_holder = socket_server_start(config.PORT);
    if (socket_holder < 0) {
        std::cerr << "Failed to start socket server\n";
        return -1;
    }

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((gridWidth_host + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (gridHeight_host + threadsPerBlock.y - 1) / threadsPerBlock.y);
    initializeGrid<<<blocksPerGrid, threadsPerBlock>>>(d_grid);
    cudaDeviceSynchronize();
    CHECK_LAST_ERROR();
    init_particles_kernel<<<1, 1>>>(particles, states);
    cudaDeviceSynchronize();
    CHECK_LAST_ERROR();

    Particle* local_particles = (Particle*)malloc(config.NUM_PARTICLES * sizeof(Particle));

    auto start = std::chrono::high_resolution_clock::now();
    int iteration = 0;

    while (1) {
        makeLinkedLists<<<numBlocks_, threadsPerBlock_>>>(d_grid, particles);
        cudaDeviceSynchronize();

        sort_single_cell_insertionSort<<<blocksPerGrid, threadsPerBlock>>>(d_grid);
        cudaDeviceSynchronize();

        check_for_collisions<<<blocksPerGrid, threadsPerBlock>>>(d_grid);
        cudaDeviceSynchronize();

        move_particles_kernel<<<numBlocks_, threadsPerBlock_>>>(particles, states);
        reset_linked_lists<<<blocksPerGrid, threadsPerBlock>>>(d_grid);
        cudaDeviceSynchronize();

        auto end = std::chrono::high_resolution_clock::now();
        iteration++;
        if (std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() >= 1000 / config.TARGET_DISPLAY_FPS) {
            cudaMemcpy(local_particles, particles, config.NUM_PARTICLES * sizeof(Particle), cudaMemcpyDeviceToHost);
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            std::cout << "Iterations per second: " << iteration / (elapsed / 1000.0) << std::endl;
            start = std::chrono::high_resolution_clock::now();
            socket_server_send(socket_holder, local_particles, config.NUM_PARTICLES * sizeof(struct Particle));
            iteration = 0;
        }
    }

    return 0;
}
