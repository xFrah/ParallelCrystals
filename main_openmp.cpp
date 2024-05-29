#include <windows.h>

#include <chrono>
#include <iostream>

extern "C" {
#include "common.h"
#include "libs/cJSON.h"
#include "libs/win_socket_server.h"
#include <omp.h>
}

typedef struct ListHead {
    Particle *head;
    omp_lock_t lock;
} ListHead;

struct Configuration config;
int gridHeight;
int gridWidth;
int cellSize;
Particle *particles;
ListHead ***grid;
int counter = 0;

void reset_linked_lists() {
    #pragma omp parallel for
    for (int i = 0; i < gridHeight; i++) {
        for (int j = 0; j < gridWidth; j++) {
            ListHead *current = grid[i][j];
            current->head = NULL;
        }
    }
}

void append_node(ListHead *listHead, Particle *newHead) {
    omp_set_lock(&(listHead->lock));
    Particle *oldHead = listHead->head;
    newHead->next_particle = oldHead;
    listHead->head = newHead;
    // counter++;
    omp_unset_lock(&(listHead->lock));
}

void make_linked_lists() {
    #pragma omp parallel for
    for (int i = 0; i < config.NUM_PARTICLES; i++) {
        int tid = omp_get_thread_num();
        int x = particles[i].x / cellSize;
        int y = particles[i].y / cellSize;
        if (x >= gridWidth) {
            x = gridWidth - 1;
        }
        if (y >= gridHeight) {
            y = gridHeight - 1;
        }
        if (x < 0) {
            x = 0;
        }
        if (y < 0) {
            y = 0;
        }
        append_node(grid[y][x], &particles[i]);
    }
}

void sort_single_cell_insertionSort() {
    const int MAX_DEPTH = config.NUM_PARTICLES + (config.NUM_PARTICLES * 0.2); // Maximum depth to prevent infinite loop

    #pragma omp parallel for
    for (int row = 0; row < gridHeight; row++) {
        for (int col = 0; col < gridWidth; col++) {
            ListHead *cell = grid[row][col];
            Particle *head = cell->head;

            if (head == NULL || head->next_particle == NULL) {
                continue; // If the cell is empty or has only one element, there's nothing to sort.
            }

            // Insertion sort on the linked list by x-coordinate
            Particle *sorted = NULL;
            Particle *current = head;
            int depth = 0; // Depth counter for cycle detection

            while (current != NULL) {
                depth++;
                if (depth > MAX_DEPTH) {
                    printf("Cycle detected or excessive depth in cell (%d, %d)\n", row, col);
                    exit(0);
                    break;
                }

                Particle *next = current->next_particle;
                if (sorted == NULL || current->x < sorted->x) {
                    current->next_particle = sorted;
                    sorted = current;
                } else {
                    Particle *search = sorted;
                    while (search->next_particle != NULL && search->next_particle->x < current->x) {
                        search = search->next_particle;
                        depth++;
                        if (depth > MAX_DEPTH) {
                            printf("Cycle detected or excessive depth in cell (%d, %d)\n", row, col);
                            exit(0);
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
            Particle *tail = sorted;
            depth = 0; // Reset depth counter for tail update
            while (tail != NULL && tail->next_particle != NULL) {
                tail = tail->next_particle;
                depth++;
                if (depth > MAX_DEPTH) {
                    printf("Cycle detected or excessive depth while updating tail in cell (%d, %d)\n", row, col);
                    exit(0);
                    break;
                }
            }
            // No need to update cell->tail, just ensure the last particle's next is null
            if (tail != NULL) {
                tail->next_particle = NULL;
            }
        }
    }
}

void check_collisions(ListHead *cell1, ListHead *cell2) {
    // parallelize this
}

void check_for_collisions() {
    // parallelize this
}

void print_linked_lists() {
#pragma omp critical
    for (int i = 0; i < gridHeight; i++) {
        for (int j = 0; j < gridWidth; j++) {
            ListHead *current = grid[i][j];
            Particle *head = current->head;
            if (head == NULL) {
                continue;
            }
            printf("Cell (%d, %d): ", i, j);
            while (head != NULL) {
                counter++;
                printf("%d -> ", head->x);
                head = head->next_particle;
            }
            printf("NULL\n");
        }
    }
}

void move_particles() {
#pragma omp parallel for
    for (int i = 0; i < config.NUM_PARTICLES; i++) {
        if (particles[i].walker) {
            particles[i].x = particles[i].x + 1 + (-2 * (rand() % 2));
            particles[i].y = particles[i].y + 1 + (-2 * (rand() % 2));
        }
    }
}

void init_particles() {
    for (int i = 0; i < config.NUM_PARTICLES; i++) {
        particles[i].id = i;
        particles[i].x = rand() % config.WIDTH;
        particles[i].y = rand() % config.HEIGHT;
        particles[i].walker = rand() % 2;
        particles[i].next_particle = NULL;
    }
    std::cout << "Initialized particles\n";
}

void initializeGrid() {
    for (int i = 0; i < gridHeight; i++) {
        for (int j = 0; j < gridWidth; j++) {
            grid[i][j] = (ListHead *)malloc(sizeof(ListHead));
            grid[i][j]->head = NULL;
            omp_init_lock(&(grid[i][j]->lock));
        }
    }
    std::cout << "Initialized grid\n";
}

void allocate_memory() {
    grid = (ListHead ***)malloc(gridHeight * sizeof(ListHead **));
    for (int i = 0; i < gridHeight; ++i) {
        grid[i] = (ListHead **)malloc(gridWidth * sizeof(ListHead *));
    }
    particles = (Particle *)malloc(config.NUM_PARTICLES * sizeof(Particle));
    std::cout << "Allocated memory for grid\n";
}

int main() {
    config = get_configuration();
    cellSize = config.CELL_SIZE;
    gridHeight = config.HEIGHT / cellSize;
    gridWidth = config.WIDTH / cellSize;

    int socket_holder = socket_server_start(config.PORT);
    if (socket_holder < 0) {
        std::cerr << "Failed to start socket server\n";
        return -1;
    }

    allocate_memory();
    init_particles();
    initializeGrid();

    auto start = std::chrono::high_resolution_clock::now();
    int iteration = 0;

// #pragma omp parallel
    {

        while (1) {
            make_linked_lists();
            // print_linked_lists();
            sort_single_cell_insertionSort();
            // check_for_collisions();
            move_particles();
            // reset_linked_lists();
            print_linked_lists();
            std::cout << "Particle counter: " << counter << std::endl;
            return 0;

// #pragma omp single
            {
                auto end = std::chrono::high_resolution_clock::now();
                iteration++;
                if (std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() >= 1000 / config.TARGET_DISPLAY_FPS) {
                    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
                    std::cout << "Iterations per second: " << iteration / (elapsed / 1000.0) << std::endl;
                    start = std::chrono::high_resolution_clock::now();
                    socket_server_send(socket_holder, particles, config.NUM_PARTICLES * sizeof(struct Particle));
                    iteration = 0;
                }
            }
        }
    }

    return 0;
}
