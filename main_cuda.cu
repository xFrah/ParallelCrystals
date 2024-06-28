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
    uint64_t particle_array_start_idx;
    uint64_t length;
};

curandState* states;
__device__ Configuration d_config;
__device__ uint64_t d_sorted = 1;
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
                // printf("Collision between particles %d and %d\n", p1->id, pk->id);
                // p1->walker = 0;
                // pk->walker = 0;
            }
            pk = pk->next_particle;
        }
        p1 = p1->next_particle;
    }
}

__global__ void makeLinkedLists(ListHead*** grid, Particle* particles) {
    uint64_t row = blockIdx.y * blockDim.y + threadIdx.y;
    uint64_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < gridHeight && col < gridWidth) {
        ListHead* cell = grid[row][col];
        uint64_t start_idx = cell->particle_array_start_idx;
        Particle* head = &particles[start_idx];
        cell->head = head;
        for (uint64_t i = start_idx; i < start_idx + cell->length - 1; i++) {
            particles[i].next_particle = &particles[i + 1];
        }
        particles[start_idx + cell->length - 1].next_particle = NULL;
    }
}

__global__ void findCellStartIndicesKernel(ListHead*** grid, Particle *d_array) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t num_threads = blockDim.x * gridDim.x;
    uint64_t start = (idx * d_config.NUM_PARTICLES) / num_threads;
    uint64_t end = ((idx + 1) * d_config.NUM_PARTICLES) / num_threads;

    for (uint64_t i = start; i < end; i++) {  // move particles
        int gridX = d_array[i].x / cellSize;
        int gridY = d_array[i].y / cellSize;

        if (gridX >= gridWidth) gridX = gridWidth - 1;
        if (gridY >= gridHeight) gridY = gridHeight - 1;
        if (gridX < 0) gridX = 0;
        if (gridY < 0) gridY = 0;

        int cellIndex = gridHeight * gridY + gridX;

        // Print the index where each cell starts
        if (i == 0 || (gridX != (d_array[i - 1].x / cellSize) && gridY != (d_array[i - 1].y / cellSize))) {
            printf("Cell %llu starts at index %d\n", cellIndex, i);
            grid[gridY][gridX]->particle_array_start_idx = i;
            if (i > 0) {
                int prevGridX = d_array[i - 1].x / cellSize;
                int prevGridY = d_array[i - 1].y / cellSize;
                int prevCellIndex = gridHeight * prevGridY + prevGridX;
                printf("Cell %d ends at index %d\n", prevCellIndex, i - 1);
                grid[prevGridY][prevGridX]->length = i - grid[prevGridY][prevGridX]->particle_array_start_idx;
            }
        }
        // TODO do the length of last cell
    }
}


__device__ bool compareParticles(Particle *a, Particle *b) {
    // Calculate grid positions for particle a
    int gridX_a = a->x / cellSize;
    int gridY_a = a->y / cellSize;
    
    if (gridX_a >= gridWidth) gridX_a = gridWidth - 1;
    if (gridY_a >= gridHeight) gridY_a = gridHeight - 1;
    if (gridX_a < 0) gridX_a = 0;
    if (gridY_a < 0) gridY_a = 0;
    
    // Calculate grid positions for particle b
    int gridX_b = b->x / cellSize;
    int gridY_b = b->y / cellSize;
    
    if (gridX_b >= gridWidth) gridX_b = gridWidth - 1;
    if (gridY_b >= gridHeight) gridY_b = gridHeight - 1;
    if (gridX_b < 0) gridX_b = 0;
    if (gridY_b < 0) gridY_b = 0;
    
    // Calculate grid indices for comparison
    int index_a = gridWidth * gridY_a + gridX_a;
    int index_b = gridWidth * gridY_b + gridX_b;
    
    // Compare by grid index, then by x in case of parity
    if (index_a == index_b) {
        return a->x > b->x;
    }
    return index_a > index_b;
}


__global__ void oddEvenSortKernel(Particle *d_array) {
    int n = d_config.NUM_PARTICLES;
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t num_threads = blockDim.x * gridDim.x;
    uint64_t start = (idx * d_config.NUM_PARTICLES) / num_threads;
    uint64_t end = ((idx + 1) * d_config.NUM_PARTICLES) / num_threads;

    for (int phase = 0; phase < (n + 1) / 2; ++phase) {
        bool local_sorted = true;

        // Odd phase
        for (uint64_t i = start; i < end; i++) {  // move particles
            if ((i % 2 == 1) && (i < n - 1)) {
                if (compareParticles(&d_array[i], &d_array[i + 1])) {
                    Particle temp = d_array[i];
                    d_array[i] = d_array[i + 1];
                    d_array[i + 1] = temp;
                    local_sorted = false;
                }
            }
        }
        __syncthreads();

        // Even phase
        for (uint64_t i = start; i < end; i++) {  // move particles
            if ((i % 2 == 0) && (i < n - 1)) {
                if (compareParticles(&d_array[i], &d_array[i + 1])) {
                    Particle temp = d_array[i];
                    d_array[i] = d_array[i + 1];
                    d_array[i + 1] = temp;
                    local_sorted = false;
                }
            }
        }
        __syncthreads();

        // If any thread in the block did a swap, mark the array as unsorted
        if (!local_sorted) {
            atomicOr(&d_sorted, 0);
        }

        // Synchronize all threads before next phase
        __syncthreads();

        // Check if the array is already sorted
        if (d_sorted == 0) {
            break;
        }
    }
}


__global__ void check_for_collisions(ListHead*** grid) {
    uint64_t row = blockIdx.y * blockDim.y + threadIdx.y;
    uint64_t col = blockIdx.x * blockDim.x + threadIdx.x;

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

__global__ void print_linked_lists(ListHead*** grid) {
    uint64_t counter = 0;

    for (uint64_t i = 0; i < gridHeight; i++) {
        for (uint64_t j = 0; j < gridWidth; j++) {
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
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t num_threads = blockDim.x * gridDim.x;
    uint64_t start = (idx * d_config.NUM_PARTICLES) / num_threads;
    uint64_t end = ((idx + 1) * d_config.NUM_PARTICLES) / num_threads;
    for (uint64_t i = start; i < end; i++) {  // move particles
        if (particles[i].walker) {
            // particles[i].x = particles[i].x + 1 + (-2 * (curand(&states[idx]) % 2));
            // particles[i].y = particles[i].y + 1 + (-2 * (curand(&states[idx]) % 2));
        }
        particles[i].next_particle = NULL;
    }
}

__global__ void init_particles_kernel(Particle* particles, curandState* states) {
    for (uint64_t i = 0; i < numBlocks_ * threadsPerBlock_; i++) {
        curand_init(d_config.SEED, i, 0, &states[i]);
    }
    for (uint64_t i = 0; i < d_config.NUM_PARTICLES; i++) {
        particles[i].id = i;
        particles[i].x = curand(&states[0]) % d_config.WIDTH;
        particles[i].y = curand(&states[0]) % d_config.HEIGHT;
        particles[i].walker = curand(&states[0]) % 2;
        particles[i].next_particle = NULL;
    }
}

__global__ void initializeGrid(ListHead*** grid) {
    uint64_t row = blockIdx.y * blockDim.y + threadIdx.y;
    uint64_t col = blockIdx.x * blockDim.x + threadIdx.x;

    uint64_t size = gridHeight * gridWidth;
    uint64_t total_size = size * sizeof(ListHead*);

    if (row == 0 && col == 0) printf("Total size: %d\n", total_size);

    if (row < gridHeight && col < gridWidth) {
        grid[row][col] = (ListHead*)malloc(sizeof(ListHead));
        if (grid[row][col] == NULL) {
            printf("Malloc failed for cell (%d, %d)\n", row, col);
            assert(0);
            return;
        }
        grid[row][col]->head = NULL;
    }
}


ListHead*** allocate_memory() {
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_config, &config, sizeof(Configuration)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(gridHeight, &gridHeight_host, sizeof(uint64_t)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(gridWidth, &gridWidth_host, sizeof(uint64_t)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(cellSize, &cellSize_host, sizeof(uint64_t)));
    CHECK_CUDA_ERROR(cudaMalloc(&particles, config.NUM_PARTICLES * sizeof(Particle)));
    CHECK_CUDA_ERROR(cudaMalloc(&states, numBlocks_ * threadsPerBlock_ * sizeof(curandState)));

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
