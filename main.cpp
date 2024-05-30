#include <windows.h>

#include <chrono>
#include <iostream>

extern "C" {
#include "common.h"
#include "libs/cJSON.h"
#include "libs/win_socket_server.h"
}

typedef struct ListHead {
    Particle *head;
} ListHead;

struct Configuration config;
int gridHeight;
int gridWidth;
int cellSize;
Particle *particles;
ListHead ***grid;

void reset_linked_lists() {
    for (int i = 0; i < gridHeight; i++) {
        for (int j = 0; j < gridWidth; j++) {
            ListHead *current = grid[i][j];
            current->head = NULL;
        }
    }
}

void append_node(ListHead *listHead, Particle *newHead) {
    Particle *oldHead = listHead->head;
    newHead->next_particle = oldHead;
    listHead->head = newHead;
}

void make_linked_lists() {
    for (int i = 0; i < config.NUM_PARTICLES; i++) {
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
    Particle *p1 = cell1->head;
    Particle *pj = cell2->head;
    int threshold = config.PARTICLE_RADIUS;

    while (p1 != NULL && pj != NULL) {
        while (pj != NULL && pj->x < p1->x - threshold) {
            pj = pj->next_particle;
        }

        Particle *pk = pj;
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

void check_for_collisions() {
    for (int row = 0; row < gridHeight; row++) {
        for (int col = 0; col < gridWidth; col++) {
            // std::cout << "Checking for collisions in cell (" << row << ", " << col << ")\n";
            ListHead *cell = grid[row][col];
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
}

void print_linked_lists() {
    for (int i = 0; i < gridHeight; i++) {
        for (int j = 0; j < gridWidth; j++) {
            ListHead *current = grid[i][j];
            Particle *head = current->head;
            if (head == NULL) {
                continue;
            }
            printf("Cell (%d, %d): ", i, j);
            while (head != NULL) {
                printf("%d -> ", head->x);
                head = head->next_particle;
            }
            printf("NULL\n");
        }
    }
}

void move_particles() {
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

    while (1) {
        make_linked_lists();
        sort_single_cell_insertionSort();
        check_for_collisions();
        move_particles();
        reset_linked_lists();

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

    return 0;
}
