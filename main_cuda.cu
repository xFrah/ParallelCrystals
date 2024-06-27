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
    curandState state;
};

__device__ Configuration d_config;
__device__ uint64_t gridHeight;
__device__ uint64_t gridWidth;
__device__ uint64_t cellSize;
struct Configuration config;
uint64_t gridHeight_host;
uint64_t gridWidth_host;
uint64_t cellSize_host;
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

__device__ void appendNode(ListHead* listHead, Particle* newHead) {
    Particle* oldHead;
    do {
        oldHead = listHead->head;          // Read the current head
        newHead->next_particle = oldHead;  // Set the new node's next to the current head
    } while (atomicCAS((unsigned long long int*)&(listHead->head), (unsigned long long int)oldHead, (unsigned long long int)newHead) != (unsigned long long int)oldHead);
}

__device__ void removeNode(ListHead* listHead, Particle* prev, Particle* current) {
    if (prev == nullptr) {
        listHead->head = current->next_particle;
    } else {
        prev->next_particle = current->next_particle;
    }
    current->next_particle = nullptr;
}

__global__ void makeLinkedLists(ListHead*** grid, Particle* particles) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t num_threads = blockDim.x * gridDim.x;
    uint64_t start = (idx * d_config.NUM_PARTICLES) / num_threads;
    uint64_t end = ((idx + 1) * d_config.NUM_PARTICLES) / num_threads;
    int64_t gridX, gridY;
    Particle* p;

    for (uint64_t i = start; i < end; i++) {
        p = &particles[i];
        gridX = p->x / cellSize;
        gridY = p->y / cellSize;

        if (gridX >= gridWidth) gridX = gridWidth - 1;
        if (gridY >= gridHeight) gridY = gridHeight - 1;
        if (gridX < 0) gridX = 0;
        if (gridY < 0) gridY = 0;
        appendNode(grid[gridY][gridX], p);
        
    }
}
__global__ void sort_single_cell_insertionSort(ListHead*** grid) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    const int MAX_DEPTH = d_config.NUM_PARTICLES + (d_config.NUM_PARTICLES * 0.2);

    if (row < gridHeight && col < gridWidth) {
        ListHead* cell = grid[row][col];
        Particle* head = cell->head;

        if (head == nullptr || head->next_particle == nullptr) {
            return;
        }

        Particle* sorted = nullptr;
        Particle* current = head;
        int depth = 0;

        while (current != nullptr) {
            if (depth++ > MAX_DEPTH) {
                printf("2) Cycle detected or excessive depth in cell (%d, %d) with %d > %d\n", row, col, depth, MAX_DEPTH);
                // assert(0);
                break;
            }

            Particle* next = current->next_particle;
            if (sorted == nullptr || current->x < sorted->x) {
                current->next_particle = sorted;
                sorted = current;
            } else {
                Particle* search = sorted;
                depth = 0;
                while (search->next_particle != nullptr && search->next_particle->x < current->x) {
                    search = search->next_particle;
                    if (depth++ > MAX_DEPTH) {
                        printf("Cycle detected or excessive depth in cell (%d, %d) with %d > %d\n", row, col, depth, MAX_DEPTH);
                        break;
                    }
                }
                current->next_particle = search->next_particle;
                search->next_particle = current;
            }
            current = next;
        }

        cell->head = sorted;

        Particle* tail = sorted;
        depth = 0;
        while (tail != nullptr && tail->next_particle != nullptr) {
            tail = tail->next_particle;
            if (depth++ > MAX_DEPTH) {
                printf("Cycle detected or excessive depth while updating tail in cell (%d, %d)\n", row, col);
                // assert(0);
                break;
            }
        }
        if (tail != nullptr) {
            tail->next_particle = nullptr;
        }
    }
}

__device__ void check_collisions(ListHead* cell1, ListHead* cell2) {
    Particle* p1 = cell1->head;
    Particle* pj = cell2->head;
    int threshold = d_config.PARTICLE_RADIUS;

    while (p1 != nullptr && pj != nullptr) {
        while (pj != nullptr && pj->x < p1->x - threshold) {
            pj = pj->next_particle;
        }

        Particle* pk = pj;
        while (pk != nullptr && pk->x <= p1->x + threshold) {
            if (p1->walker != pk->walker && abs(p1->y - pk->y) <= threshold) {
                // printf("Collision between particles %d and %d\n", p1->id, pk->id);
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

__device__ void print_linked_lists(ListHead*** grid) {
    int counter = 0;

    for (int i = 0; i < gridHeight; i++) {
        for (int j = 0; j < gridWidth; j++) {
            ListHead* cell = grid[i][j];
            if (cell->head == nullptr) {
                // printf("Cell (%d, %d) is empty\n", i, j);
                continue;
            }
            Particle* p = cell->head;
            printf("[%d] Cell (%d, %d): ", counter, i, j);
            while (p != nullptr) {
                printf("%d (%d, %d) ", p->id, p->x, p->y);
                counter++;
                p = p->next_particle;
            }
            printf("\n");
        }
    }
}

__global__ void move_particles_kernel(ListHead*** grid) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row * gridWidth + col;
    int newGridX, newGridY;
    int i = 0;
    
    if (row < gridHeight && col < gridWidth) {
        ListHead* cell = grid[row][col];
        Particle* p = cell->head;
        Particle* prev = nullptr;
        Particle* next_p;
        bool removed;
        while (p != nullptr) {
            removed = false;
            next_p = p->next_particle;
            if (p->walker) {
                p->x = p->x + 1 + (-2 * (curand(&cell->state) % 2));
                p->y = p->y + 1 + (-2 * (curand(&cell->state) % 2));

                newGridX = p->x / cellSize;
                newGridY = p->y / cellSize;

                if (newGridX >= gridWidth) newGridX = gridWidth - 1;
                if (newGridY >= gridHeight) newGridY = gridHeight - 1;
                if (newGridX < 0) newGridX = 0;
                if (newGridY < 0)newGridY = 0;

                if (newGridX != col || newGridY != row) {
                    removeNode(cell, prev, p);
                    appendNode(grid[newGridY][newGridX], p);
                    removed = true;
                }
            }
            if (removed == false) {
                prev = p;
            }
            p = next_p;
            i++;
            if (i > 1000) {
                // print_linked_lists(grid);
                printf("Cycle detected in cell (%d, %d)\n", row, col);
                // check if cell or its components are null or nullpointer
                if (cell == nullptr) printf("Cell is null\n");

                if (cell->head == nullptr) {
                    printf("Cell head is null\n");
                } else {
                    printf("Cell: %d, %d\n", cell->head->id, cell->head->walker);
                }

                if (p == nullptr) {
                    printf("P is null\n");
                } else {
                    printf("P: %d, %d\n", p->id, p->walker);
                }
                if (prev == nullptr) {
                    printf("Prev is null\n");
                } else {
                    printf("Prev: %d, %d\n", prev->id, prev->walker);
                }
                assert(0);
            }
        }
    }
}

__global__ void init_particles_kernel(Particle* particles) {
    curandState state;
    curand_init(d_config.SEED, 0, 0, &state);
    for (uint64_t i = 0; i < d_config.NUM_PARTICLES; i++) {
        particles[i].id = i;
        particles[i].x = curand(&state) % d_config.WIDTH;
        particles[i].y = curand(&state) % d_config.HEIGHT;
        particles[i].walker = curand(&state) % 2;
        particles[i].next_particle = nullptr;
    }
}

__global__ void initializeGrid(ListHead*** grid) {
    uint64_t row = blockIdx.y * blockDim.y + threadIdx.y;
    uint64_t col = blockIdx.x * blockDim.x + threadIdx.x;

    uint64_t size = gridHeight * gridWidth;
    uint64_t total_size = size * sizeof(ListHead*);

    if (row == 0 && col == 0) printf("Total size: %d\n", total_size);

    int idx = row * gridWidth + col;

    if (row < gridHeight && col < gridWidth) {
        grid[row][col] = (ListHead*)malloc(sizeof(ListHead));
        if (grid[row][col] == NULL) {
            printf("Malloc failed for cell (%d, %d)\n", row, col);
            assert(0);
            return;
        }
        grid[row][col]->head = nullptr;
        curand_init(d_config.SEED, idx, 0, &grid[row][col]->state);
    }
}


ListHead*** allocate_memory() {
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_config, &config, sizeof(Configuration)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(gridHeight, &gridHeight_host, sizeof(uint64_t)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(gridWidth, &gridWidth_host, sizeof(uint64_t)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(cellSize, &cellSize_host, sizeof(uint64_t)));
    CHECK_CUDA_ERROR(cudaMalloc(&particles, config.NUM_PARTICLES * sizeof(Particle)));

    std::cout << "Grid size: " << gridHeight_host << "x" << gridWidth_host << std::endl;

    ListHead*** h_grid = (ListHead***)malloc(gridHeight_host * sizeof(ListHead**));
    if (h_grid == nullptr) {
        std::cerr << "Error: Failed to allocate host memory for grid rows" << std::endl;
        exit(1);
    }
    for (uint64_t i = 0; i < gridHeight_host; ++i) {
        h_grid[i] = (ListHead**)malloc(gridWidth_host * sizeof(ListHead*));
        if (h_grid[i] == nullptr) {
            std::cerr << "Error: Failed to allocate host memory for grid columns at row " << i << std::endl;
            exit(1);
        }
    }

    std::cout << "Allocated memory for the grid on the host" << std::endl;

    ListHead*** d_grid;
    CHECK_CUDA_ERROR(cudaMalloc(&d_grid, gridHeight_host * sizeof(ListHead**)));
    for (uint64_t i = 0; i < gridHeight_host; ++i) {
        ListHead** d_row;
        CHECK_CUDA_ERROR(cudaMalloc(&d_row, gridWidth_host * sizeof(ListHead*)));
        CHECK_CUDA_ERROR(cudaMemcpy(&d_grid[i], &d_row, sizeof(ListHead*), cudaMemcpyHostToDevice));
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

    int socket_holder;
    if (config.SHOW_VISUALLY) {
        std::cout << "Starting socket server on port " << config.PORT << "\n";
        socket_holder = socket_server_start(config.PORT);
        if (socket_holder < 0) {
            std::cerr << "Failed to start socket server\n";
            return -1;
        }
    }

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((gridWidth_host + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (gridHeight_host + threadsPerBlock.y - 1) / threadsPerBlock.y);
    initializeGrid<<<blocksPerGrid, threadsPerBlock>>>(d_grid);
    cudaDeviceSynchronize();
    CHECK_LAST_ERROR();
    init_particles_kernel<<<1, 1>>>(particles);
    cudaDeviceSynchronize();
    CHECK_LAST_ERROR();

    makeLinkedLists<<<numBlocks_, threadsPerBlock_>>>(d_grid, particles);
    cudaDeviceSynchronize();
    CHECK_LAST_ERROR();

    Particle* local_particles = (Particle*)malloc(config.NUM_PARTICLES * sizeof(Particle));

    auto start = std::chrono::high_resolution_clock::now();
    int iteration = 0;

    while (1) {
        sort_single_cell_insertionSort<<<blocksPerGrid, threadsPerBlock>>>(d_grid);
        cudaDeviceSynchronize();
        CHECK_LAST_ERROR();

        check_for_collisions<<<blocksPerGrid, threadsPerBlock>>>(d_grid);
        cudaDeviceSynchronize();
        CHECK_LAST_ERROR();
        
        // print_linked_lists<<<1, 1>>>(d_grid);

        move_particles_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_grid);
        cudaDeviceSynchronize();
        CHECK_LAST_ERROR();

        auto end = std::chrono::high_resolution_clock::now();
        iteration++;
        if (std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() >= 1000 / config.TARGET_DISPLAY_FPS) {
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            std::cout << "Iterations per second: " << iteration / (elapsed / 1000.0) << std::endl;
            start = std::chrono::high_resolution_clock::now();
            if (config.SHOW_VISUALLY) {
                cudaMemcpy(local_particles, particles, config.NUM_PARTICLES * sizeof(Particle), cudaMemcpyDeviceToHost);
                socket_server_send(socket_holder, local_particles, config.NUM_PARTICLES * sizeof(struct Particle));
            }
            iteration = 0;
        }
    }

    return 0;
}
