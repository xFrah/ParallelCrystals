#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <windows.h>

#include <chrono>
#include <iostream>

// parametri ottimali per la nostra 2080ti
#define numBlocks_ 68
#define threadsPerBlock_ 32

extern "C" {
#include "common.h"                 // codice condiviso tra le varie implementazioni
#include "libs/cJSON.h"             // libreria per il parsing di file json
#include "libs/win_socket_server.h" // libreria per la comunicazione via socket
}

/* Testa delle linked list delle particelle */
struct ListHead {
    Particle *head;
};

/* Array di stati curand. Necessario per randomizzazione migliore */
curandState *states;
/* Configurazione memorizzata sul device */
__device__ Configuration d_config;
/* Altezza della griglia memorizzata sul device */
__device__ uint64_t gridHeight;
/* Larghezza della griglia memorizzata sul device */
__device__ uint64_t gridWidth;
/* Dimensione del lato di una cella della griglia */
__device__ uint64_t cellSize;
/* Configurazione memorizzata sull'host */
struct Configuration config;
/* Altezza della griglia memorizzata sull'host */
uint64_t gridHeight_host;
/* Larghezza della griglia memorizzata sull'host */
uint64_t gridWidth_host;
/* Dimensione del lato di una cella memorizzata sull'host */
uint64_t cellSize_host;
/* Array di particelle */
Particle *particles;

// per debuggare le chiamate al framework cuda
#define CHECK_CUDA_ERROR(call)                                                \
    {                                                                         \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA error in " << __FILE__ << " at line "          \
                      << __LINE__ << ": " << cudaGetErrorString(err) << "\n"; \
            exit(err);                                                        \
        }                                                                     \
    }

// per debuggare le chiamate al framework cuda
#define CHECK_LAST_ERROR()                                                    \
    {                                                                         \
        cudaError_t err = cudaGetLastError();                                 \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA error in " << __FILE__ << " at line "          \
                      << __LINE__ << ": " << cudaGetErrorString(err) << "\n"; \
            exit(err);                                                        \
        }                                                                     \
    }

/*
 * Questo kernel resetta le linked list di tutte le celle della griglia.
 */
__global__ void reset_linked_lists(ListHead ***grid) {
    uint64_t row = blockIdx.y * blockDim.y + threadIdx.y; // calcolo cella da processare
    uint64_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < gridHeight && col < gridWidth) { // se la cella è nei limiti della griglia
        ListHead *cell = grid[row][col];
        cell->head = NULL;
    }
}

/*
 * Questa funzione inserisce una particella in testa alla linked list.
 * In questa implementazione parallela, sono necessarie operazioni atomiche.
 */
__device__ void appendNode(ListHead *listHead, Particle *newHead) {
    Particle *oldHead;
    do {
        oldHead = listHead->head;         // salviamo la vecchia testa
        newHead->next_particle = oldHead; // settiamo il next della nuova testa alla vecchia testa
    } while (atomicCAS((unsigned long long int *)&(listHead->head), (unsigned long long int)oldHead, (unsigned long long int)newHead) != (unsigned long long int)oldHead);
    // atomic check and set per cambiare atomicamente la testa alla linked list.
}

/*
 * Questo kernel inserisce le particelle nelle linked list delle celle della griglia.
 * Ogni particella viene inserita nella linked list della cella in cui si trova.
 */
__global__ void makeLinkedLists(ListHead ***grid, Particle *particles) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t num_threads = blockDim.x * gridDim.x;
    // calcolo della porzione dell'array da processare
    uint64_t start = (idx * d_config.NUM_PARTICLES) / num_threads;
    int64_t end = ((idx + 1) * d_config.NUM_PARTICLES) / num_threads;
    int64_t gridX, gridY;
    Particle *p;

    for (uint64_t i = start; i < end; i++) {
        p = &particles[i];
        gridX = p->x / cellSize; // calcoliamo la cella in cui si trova la particella
        gridY = p->y / cellSize;

        if (gridX >= gridWidth) { // controlliamo che le particelle non escano dalla griglia
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
        appendNode(grid[gridY][gridX], p); // inseriamo la particella nella linked list della cella corrispondente
    }
}

/*
 * Questo kernel ordina le linked list delle singole celle della griglia.
 * Utilizza l'insertion sort come algoritmo di ordinamento.
 */
__global__ void sort_single_cell_insertionSort(ListHead ***grid) {
    uint64_t row = blockIdx.y * blockDim.y + threadIdx.y; // calcolo cella da processare
    uint64_t col = blockIdx.x * blockDim.x + threadIdx.x;

    // profondità massima per evitare cicli infiniti
    const int MAX_DEPTH = d_config.NUM_PARTICLES + (d_config.NUM_PARTICLES * 0.2);

    if (row < gridHeight && col < gridWidth) { // controlliamo se la cella esiste
        ListHead *cell = grid[row][col];
        Particle *head = cell->head;

        // se la cella è vuota o contiene una sola particella, non c'è bisogno di ordinare
        if (head == NULL || head->next_particle == NULL) {
            return;
        }

        Particle *sorted = NULL;
        Particle *current = head;
        int depth = 0;

        while (current != NULL) {
            if (depth++ > MAX_DEPTH) { // controlliamo se siamo in un ciclo infinito
                printf("2) Cycle detected or excessive depth in cell (%d, %d) with %d > %d\n", row, col, depth, MAX_DEPTH);
                // assert(0);
                break;
            }

            Particle *next = current->next_particle;
            if (sorted == NULL || current->x < sorted->x) {
                current->next_particle = sorted;
                sorted = current;
            } else {
                // troviamo il posto giusto per la particella corrente
                Particle *search = sorted;
                depth = 0;
                while (search->next_particle != NULL && search->next_particle->x < current->x) {
                    search = search->next_particle;
                    // controlliamo se siamo in un ciclo infinito
                    if (depth++ > MAX_DEPTH) {
                        printf("Cycle detected or excessive depth in cell (%d, %d) with %d > %d\n", row, col, depth, MAX_DEPTH);
                        break;
                    }
                }
                // inseriamo la particella corrente nella posizione corretta
                current->next_particle = search->next_particle;
                search->next_particle = current;
            }
            current = next;
        }

        cell->head = sorted;

        Particle *tail = sorted;
        depth = 0;
        // troviamo la coda della linked list e la settiamo a NULL
        while (tail != NULL && tail->next_particle != NULL) {
            tail = tail->next_particle;
            // controlliamo se siamo in un ciclo infinito
            if (depth++ > MAX_DEPTH) {
                printf("Cycle detected or excessive depth while updating tail in cell (%d, %d)\n", row, col);
                // assert(0);
                break;
            }
        }
        if (tail != NULL) {
            tail->next_particle = NULL;
        }
    }
}

/*
 * Questo kernel controlla le collisioni tra due celle.
 * Per ogni particella nella prima cella, controlla se collide con le particelle nella seconda cella.
 * Se due particelle collidono, le fermiamo.
 */
__device__ void check_collisions(ListHead *cell1, ListHead *cell2) {
    Particle *p1 = cell1->head;
    Particle *pj = cell2->head;
    int threshold = d_config.PARTICLE_RADIUS;

    while (p1 != NULL && pj != NULL) {
        while (pj != NULL && pj->x < p1->x - threshold) { // scartiamo le particelle troppo a sinistra
            pj = pj->next_particle;
        }

        Particle *pk = pj;
        while (pk != NULL && pk->x <= p1->x + threshold) { // controlliamo le particelle nella cella adiacente
            // controlliamo se le particelle collidono
            if (p1->walker != pk->walker && abs(p1->y - pk->y) <= threshold) {
                // se collidono, le fermiamo
                p1->walker = 0;
                pk->walker = 0;
            }
            pk = pk->next_particle;
        }
        p1 = p1->next_particle;
    }
}

/*
 * Questo kernel controlla le collisioni tra le particelle in tempo lineare.
 * Per ogni cella della griglia, controlla le collisioni con le celle adiacenti.
 * Se due particelle collidono, si fermano.
 */
__global__ void check_for_collisions(ListHead ***grid) {
    uint64_t row = blockIdx.y * blockDim.y + threadIdx.y;
    uint64_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < gridHeight && col < gridWidth) {
        ListHead *cell = grid[row][col];
        // individuiamo quale pattern dobbiamo seguire per controllare le collisioni
        if (row == 0 && col == 0) {
            // angolo in alto a sinistra: confronto con cella a destra, cella sotto, cella in basso a destra
            check_collisions(cell, grid[row][col + 1]);     // cella a destra
            check_collisions(cell, grid[row + 1][col]);     // cella sotto
            check_collisions(cell, grid[row + 1][col + 1]); // cella in basso a destra
            check_collisions(cell, cell);                   // confronto con se stessa
        } else if (row == 0 && col == gridWidth - 1) {
            // angolo in alto a destra: confronto con cella sotto e cella in basso a sinistra
            check_collisions(cell, grid[row + 1][col]);     // cella sotto
            check_collisions(cell, grid[row + 1][col - 1]); // cella in basso a sinistra
            check_collisions(cell, cell);                   // confronto con se stessa
        } else if (row == gridHeight - 1 && col == 0) {
            // angolo in basso a sinistra: confronto con cella a destra
            check_collisions(cell, grid[row][col + 1]); // cella a destra
            check_collisions(cell, cell);               // confronto con se stessa
        } else if (row == gridHeight - 1 && col == gridWidth - 1) {
            // angolo in basso a destra: confronto con nessuno
            check_collisions(cell, cell); // confronto con se stessa
        } else if (row == 0) {
            // bordo superiore: confronto con cella a destra, cella in basso a sinistra, cella sotto e cella in basso a destra
            check_collisions(cell, grid[row][col + 1]);     // cella a destra
            check_collisions(cell, grid[row + 1][col - 1]); // cella in basso a sinistra
            check_collisions(cell, grid[row + 1][col]);     // cella sotto
            check_collisions(cell, grid[row + 1][col + 1]); // cella in basso a destra
            check_collisions(cell, cell);                   // confronto con se stessa
        } else if (row == gridHeight - 1) {
            // bordo inferiore: confronto con cella a destra
            check_collisions(cell, grid[row][col + 1]); // cella a destra
            check_collisions(cell, cell);               // confronto con se stessa
        } else if (col == 0) {
            // bordo sinistro: confronto con cella a destra, cella sotto, cella in basso a destra
            check_collisions(cell, grid[row][col + 1]);     // cella a destra
            check_collisions(cell, grid[row + 1][col]);     // cella sotto
            check_collisions(cell, grid[row + 1][col + 1]); // cella in basso a destra
            check_collisions(cell, cell);                   // confronto con se stessa
        } else if (col == gridWidth - 1) {
            // bordo destro: confronto con cella sotto e cella in basso a sinistra
            check_collisions(cell, grid[row + 1][col]);     // cella sotto
            check_collisions(cell, grid[row + 1][col - 1]); // cella in basso a sinistra
            check_collisions(cell, cell);
        } else {
            // centro: confronto con cella a destra, cella sotto, cella in basso a destra, cella in basso a sinistra
            check_collisions(cell, grid[row][col + 1]);     // cella a destra
            check_collisions(cell, grid[row + 1][col]);     // cella sotto
            check_collisions(cell, grid[row + 1][col + 1]); // cella in basso a destra
            check_collisions(cell, grid[row + 1][col - 1]); // cella in basso a sinistra
            check_collisions(cell, cell);                   // confronto con se stessa
        }
    }
}

/*
 * Kernel di debug per stampare le linked list di ogni cella della griglia.
 */
__global__ void print_linked_lists(ListHead ***grid) {
    uint64_t counter = 0;

    for (uint64_t i = 0; i < gridHeight; i++) { // iteriamo su tutte le celle della griglia
        for (uint64_t j = 0; j < gridWidth; j++) {
            ListHead *cell = grid[i][j];
            if (cell->head == NULL) {
                continue;
            }
            Particle *p = cell->head;
            printf("[%d] Cell (%d, %d): ", counter, i, j);
            while (p != NULL) { // stampiamo la cella corrente iterando sulla linked list
                printf("%d (%d, %d) ", p->id, p->x, p->y);
                counter++;
                p = p->next_particle;
            }
            printf("\n");
        }
    }
}

/*
 * Questo kernel muove le particelle di 1 pixel in una direzione casuale.
 */
__global__ void move_particles_kernel(Particle *particles, curandState *states) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t num_threads = blockDim.x * gridDim.x;
    uint64_t start = (idx * d_config.NUM_PARTICLES) / num_threads;
    uint64_t end = ((idx + 1) * d_config.NUM_PARTICLES) / num_threads;
    for (uint64_t i = start; i < end; i++) { // move particles
        if (particles[i].walker) {
            particles[i].x = particles[i].x + 1 + (-2 * (curand(&states[idx]) % 2));
            particles[i].y = particles[i].y + 1 + (-2 * (curand(&states[idx]) % 2));
        }
        particles[i].next_particle = NULL;
    }
}

/*
 * Questo kernel inizializza le particelle con valori casuali.
 */
__global__ void init_particles_kernel(Particle *particles, curandState *states) {
    for (uint64_t i = 0; i < numBlocks_ * threadsPerBlock_; i++) {
        curand_init(d_config.SEED, i, 0, &states[i]);
    }
    for (uint64_t i = 0; i < d_config.NUM_PARTICLES; i++) {
        particles[i].id = i;
        particles[i].x = curand(&states[0]) % d_config.WIDTH;
        particles[i].y = curand(&states[0]) % d_config.HEIGHT;
        particles[i].walker = curand(&states[0]) % 2; // flag per indicare se la particella si muove
        particles[i].next_particle = NULL;            // necessario per le linked list
    }
}

/*
 * Questo kernel inizializza la griglia con le linked list vuote.
 * La griglia ha dimensioni gridHeight x gridWidth.
 */
__global__ void initializeGrid(ListHead ***grid) {
    uint64_t row = blockIdx.y * blockDim.y + threadIdx.y; // calcolo cella da processare
    uint64_t col = blockIdx.x * blockDim.x + threadIdx.x;

    uint64_t size = gridHeight * gridWidth;
    uint64_t total_size = size * sizeof(ListHead *);

    if (row == 0 && col == 0)
        printf("Total size: %d\n", total_size);

    if (row < gridHeight && col < gridWidth) {                 // se la cella è nei limiti della griglia
        grid[row][col] = (ListHead *)malloc(sizeof(ListHead)); // allochiamo memoria per la cella
        if (grid[row][col] == NULL) {
            printf("Malloc failed for cell (%d, %d)\n", row, col);
            assert(0);
            return;
        }
        grid[row][col]->head = NULL;
    }
}

/*
 * Questa funzione alloca la memoria per le particelle e per la griglia.
 */
ListHead ***allocate_memory() {
    // passiamo le variabili globali alla memoria del device
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_config, &config, sizeof(Configuration)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(gridHeight, &gridHeight_host, sizeof(uint64_t)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(gridWidth, &gridWidth_host, sizeof(uint64_t)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(cellSize, &cellSize_host, sizeof(uint64_t)));
    // allochiamo la memoria per le particelle e per gli stati curand sul device
    CHECK_CUDA_ERROR(cudaMalloc(&particles, config.NUM_PARTICLES * sizeof(Particle)));
    CHECK_CUDA_ERROR(cudaMalloc(&states, numBlocks_ * threadsPerBlock_ * sizeof(curandState)));

    std::cout << "Grid size: " << gridHeight_host << "x" << gridWidth_host << std::endl;

    // allochiamo la memoria per la griglia sull'host
    ListHead ***h_grid = (ListHead ***)malloc(gridHeight_host * sizeof(ListHead **));
    if (h_grid == nullptr) {
        std::cerr << "Error: Failed to allocate host memory for grid rows" << std::endl;
        exit(1);
    }
    for (uint64_t i = 0; i < gridHeight_host; ++i) {
        h_grid[i] = (ListHead **)malloc(gridWidth_host * sizeof(ListHead *));
        if (h_grid[i] == nullptr) {
            std::cerr << "Error: Failed to allocate host memory for grid columns at row " << i << std::endl;
            exit(1);
        }
    }

    std::cout << "Allocated memory for the grid on the host" << std::endl;

    // allochiamo la memoria per la griglia sul device
    ListHead ***d_grid;
    CHECK_CUDA_ERROR(cudaMalloc(&d_grid, gridHeight_host * sizeof(ListHead **)));
    for (uint64_t i = 0; i < gridHeight_host; ++i) {
        ListHead **d_row;
        CHECK_CUDA_ERROR(cudaMalloc(&d_row, gridWidth_host * sizeof(ListHead *)));
        CHECK_CUDA_ERROR(cudaMemcpy(&d_grid[i], &d_row, sizeof(ListHead *), cudaMemcpyHostToDevice));
    }

    std::cout << "Allocated memory for the grid on the device" << std::endl;
    return d_grid;
}

int main() {
    config = get_configuration(); // prendiamo la configurazione dal file json
    cellSize_host = config.CELL_SIZE;
    gridHeight_host = config.HEIGHT / cellSize_host;
    gridWidth_host = config.WIDTH / cellSize_host;

    ListHead ***d_grid = allocate_memory(); // allochiamo la memoria per le varie strutture

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

    // array di particelle sull'host per la comunicazione via socket
    Particle *local_particles = (Particle *)malloc(config.NUM_PARTICLES * sizeof(Particle));

    auto start = std::chrono::high_resolution_clock::now(); // variabili per il calcolo della performance
    int iteration = 0;

    while (1) {
        // inseriamo le particelle nelle linked list della griglia
        makeLinkedLists<<<numBlocks_, threadsPerBlock_>>>(d_grid, particles);
        cudaDeviceSynchronize();

        // ordiniamo le linked list
        sort_single_cell_insertionSort<<<blocksPerGrid, threadsPerBlock>>>(d_grid);
        cudaDeviceSynchronize();

        // controlliamo le collisioni in tempo lineare
        check_for_collisions<<<blocksPerGrid, threadsPerBlock>>>(d_grid);
        cudaDeviceSynchronize();

        // muoviamo le particelle di 1 pixel in una direzione casuale
        move_particles_kernel<<<numBlocks_, threadsPerBlock_>>>(particles, states);
        // resettiamo le linked list per il prossimo ciclo
        reset_linked_lists<<<blocksPerGrid, threadsPerBlock>>>(d_grid);
        cudaDeviceSynchronize();

        auto end = std::chrono::high_resolution_clock::now(); // calcoliamo la performance
        iteration++;
        if (std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() >= 1000 / config.TARGET_DISPLAY_FPS) {
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            std::cout << "Iterations per second: " << iteration / (elapsed / 1000.0) << std::endl;
            start = std::chrono::high_resolution_clock::now();
            if (config.SHOW_VISUALLY) {
                // copiamo le particelle dal device all'host
                cudaMemcpy(local_particles, particles, config.NUM_PARTICLES * sizeof(Particle), cudaMemcpyDeviceToHost);
                // inviamo le particelle via socket all'interfaccia grafica
                socket_server_send(socket_holder, local_particles, config.NUM_PARTICLES * sizeof(struct Particle));
            }
            iteration = 0;
        }
    }

    return 0;
}
