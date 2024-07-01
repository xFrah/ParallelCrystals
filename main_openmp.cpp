#include <windows.h>

#include <chrono> // per il calcolo della performance
#include <iostream>

extern "C" {
#include "common.h"                 // codice condiviso tra le varie implementazioni
#include "libs/cJSON.h"             // libreria per il parsing di file json
#include "libs/win_socket_server.h" // libreria per la comunicazione via socket
#include <omp.h>                    // libreria per il supporto alle direttive OpenMP
}

typedef struct ListHead {
    Particle *head;
    omp_lock_t lock; // Lock per l'inserimento nella linked list
} ListHead;

/* Configurazione del programma */
struct Configuration config;
/* Altezza della griglia in celle */
int gridHeight;
/* Larghezza della griglia in celle */
int gridWidth;
/* Dimensione del lato di una cella della griglia */
int cellSize;
/* Array di particelle */
Particle *particles;
/* Griglia di linked list */
ListHead ***grid;

/*
 * Questa funzione resetta le linked list di tutte le celle della griglia.
 */
void reset_linked_lists() {
#pragma omp for
    for (int i = 0; i < gridHeight; i++) { // iteriamo su tutte le celle della griglia
        for (int j = 0; j < gridWidth; j++) {
            ListHead *current = grid[i][j];
            current->head = NULL; // resettiamo la testa della linked list
        }
    }
}

/*
 * Questa funzione inserisce una particella in testa alla linked list.
 * @param listHead: puntatore ala linked list
 * @param newHead: nuova testa della linked list
 */
void append_node(ListHead *listHead, Particle *newHead) {
    omp_set_lock(&(listHead->lock));    // acquisiamo il lock per l'inserimento
    Particle *oldHead = listHead->head; // salviamo la vecchia testa
    newHead->next_particle = oldHead;   // settiamo il next della nuova testa alla vecchia testa
    listHead->head = newHead;           // settiamo la nuova testa
    omp_unset_lock(&(listHead->lock));  // rilasciamo il lock
}

/*
 * Questa funzione inserisce le particelle nelle linked list delle celle della griglia.
 * Ogni particella viene inserita nella linked list della cella in cui si trova.
 */
void make_linked_lists() {
#pragma omp for
    for (int i = 0; i < config.NUM_PARTICLES; i++) {
        int x = particles[i].x / cellSize; // calcoliamo la cella in cui si trova la particella
        int y = particles[i].y / cellSize;
        if (x >= gridWidth) { // controlliamo che le particelle non escano dalla griglia
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
        append_node(grid[y][x], &particles[i]); // inseriamo la particella nella linked list della cella corrispondente
    }
}

/*
 * Questa funzione ordina le linked list delle singole celle della griglia.
 * Utilizza l'insertion sort come algoritmo di ordinamento.
 */
void sort_single_cell_insertionSort() {
    // profondità massima per evitare cicli infiniti
    const int MAX_DEPTH = config.NUM_PARTICLES + (config.NUM_PARTICLES * 0.2); // Maximum depth to prevent infinite loop

#pragma omp for
    for (int row = 0; row < gridHeight; row++) { // iteriamo su tutte le celle della griglia
        for (int col = 0; col < gridWidth; col++) {
            ListHead *cell = grid[row][col];
            Particle *head = cell->head;

            // se la cella è vuota o contiene una sola particella, non c'è bisogno di ordinare
            if (head == NULL || head->next_particle == NULL) {
                continue;
            }

            Particle *sorted = NULL;
            Particle *current = head;
            int depth = 0;

            while (current != NULL) {
                depth++;
                if (depth > MAX_DEPTH) { // controlliamo se siamo in un ciclo infinito
                    printf("Cycle detected or excessive depth in cell (%d, %d)\n", row, col);
                    exit(0);
                    break;
                }

                Particle *next = current->next_particle;
                if (sorted == NULL || current->x < sorted->x) {
                    current->next_particle = sorted;
                    sorted = current;
                } else {
                    // troviamo il posto giusto per la particella corrente
                    Particle *search = sorted;
                    while (search->next_particle != NULL && search->next_particle->x < current->x) {
                        search = search->next_particle;
                        depth++;
                        if (depth > MAX_DEPTH) { // controlliamo se siamo in un ciclo infinito
                            printf("Cycle detected or excessive depth in cell (%d, %d)\n", row, col);
                            exit(0);
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
                depth++;
                // controlliamo se siamo in un ciclo infinito
                if (depth > MAX_DEPTH) {
                    printf("Cycle detected or excessive depth while updating tail in cell (%d, %d)\n", row, col);
                    exit(0);
                    break;
                }
            }
            if (tail != NULL) {
                tail->next_particle = NULL;
            }
        }
    }
}

/*
 * Questa funzione controlla le collisioni tra due celle.
 * Per ogni particella nella prima cella, controlla se collide con le particelle nella seconda cella.
 * Se due particelle collidono, le fermiamo.
 * @param cell1: puntatore alla prima cella
 * @param cell2: puntatore alla seconda cella
 */
void check_collisions(ListHead *cell1, ListHead *cell2) {
    Particle *p1 = cell1->head;
    Particle *pj = cell2->head;
    int threshold = config.PARTICLE_RADIUS;

    while (p1 != NULL && pj != NULL) {
        while (pj != NULL && pj->x < p1->x - threshold) { // scartiamo le particelle troppo a sinistra
            pj = pj->next_particle;
        }

        Particle *pk = pj;
        while (pk != NULL && pk->x <= p1->x + threshold) {
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
 * Questa funzione controlla le collisioni tra le particelle in tempo lineare.
 * Per ogni cella della griglia, controlla le collisioni con le celle adiacenti.
 * Se due particelle collidono, si fermano.
 */
void check_for_collisions() {
#pragma omp for
    for (int row = 0; row < gridHeight; row++) { // iteriamo su tutte le celle della griglia
        for (int col = 0; col < gridWidth; col++) {
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
}

/*
 * Funzione di debug per stampare le linked list di ogni cella della griglia.
 */
void print_linked_lists() {
#pragma omp critical
    for (int i = 0; i < gridHeight; i++) { // iteriamo su tutte le celle della griglia
        for (int j = 0; j < gridWidth; j++) {
            ListHead *current = grid[i][j];
            Particle *head = current->head;
            if (head == NULL) {
                continue;
            }
            printf("Cell (%d, %d): ", i, j);
            while (head != NULL) { // stampiamo la cella corrente iterando sulla linked list
                printf("%d -> ", head->x);
                head = head->next_particle;
            }
            printf("NULL\n");
        }
    }
}

/*
 * Questa funzione muove le particelle di 1 pixel in una direzione casuale.
 */
void move_particles() {
#pragma omp for
    for (int i = 0; i < config.NUM_PARTICLES; i++) {
        // se la particella è un walker, la muoviamo
        if (particles[i].walker) {
            // muoviamo la particella di 1 pixeL
            particles[i].x = particles[i].x + 1 + (-2 * (rand() % 2));
            particles[i].y = particles[i].y + 1 + (-2 * (rand() % 2));
        }
    }
}

/*
 * Questa funzione inizializza le particelle con valori casuali.
 */
void init_particles() {
    for (int i = 0; i < config.NUM_PARTICLES; i++) {
        particles[i].id = i;
        particles[i].x = rand() % config.WIDTH;
        particles[i].y = rand() % config.HEIGHT;
        particles[i].walker = rand() % 2;  // flag per indicare se la particella si muove
        particles[i].next_particle = NULL; // necessario per le linked list
    }
    std::cout << "Initialized particles\n";
}

/*
 * Questa funzione inizializza la griglia con le linked list vuote.
 * La griglia ha dimensioni gridHeight x gridWidth.
 */
void initializeGrid() {
    for (int i = 0; i < gridHeight; i++) {
        for (int j = 0; j < gridWidth; j++) {
            grid[i][j] = (ListHead *)malloc(sizeof(ListHead)); // allochiamo la memoria per la testa della linked list
            grid[i][j]->head = NULL;                           // inizializziamo la testa della linked list a NULL
            omp_init_lock(&(grid[i][j]->lock));                // inizializziamo il lock necessario per l'inserimento nelle celle
        }
    }
    std::cout << "Initialized grid\n";
}

/*
 * Questa funzione alloca la memoria per le particelle e per la griglia.
 */
void allocate_memory() {
    grid = (ListHead ***)malloc(gridHeight * sizeof(ListHead **));
    for (int i = 0; i < gridHeight; ++i) {
        grid[i] = (ListHead **)malloc(gridWidth * sizeof(ListHead *));
    }
    particles = (Particle *)malloc(config.NUM_PARTICLES * sizeof(Particle));
    std::cout << "Allocated memory for grid\n";
}

int main() {
    config = get_configuration(); // prendiamo la configurazione dal file json
    cellSize = config.CELL_SIZE;
    gridHeight = config.HEIGHT / cellSize;
    gridWidth = config.WIDTH / cellSize;

    if (config.USE_MAX_HARDWARE_THREADS) { // usiamo il numero massimo di thread disponibili
        int num_threads = omp_get_max_threads();
        std::cout << "Using maximum hardware threads: " << num_threads << "\n";
        omp_set_num_threads(num_threads);
    } else {
        omp_set_num_threads(config.NUM_THREADS);
        std::cout << "Using " << config.NUM_THREADS << " threads\n";
    }

    int socket_holder; // inizializziamo la connessione all'interfaccia grafica
    if (config.SHOW_VISUALLY) {
        std::cout << "Starting socket server on port " << config.PORT << "\n";
        socket_holder = socket_server_start(config.PORT);
        if (socket_holder < 0) {
            std::cerr << "Failed to start socket server\n";
            return -1;
        }
    }

    allocate_memory(); // allochiamo la memoria per le particelle e per la griglia
    init_particles();  // inizializziamo l'array di particelle e riempiamo i campi degli struct
    initializeGrid();  // inizializziamo la griglia

    auto start = std::chrono::high_resolution_clock::now(); // variabili per il calcolo della performance
    int iteration = 0;

#pragma omp parallel
    {

        while (1) {
            make_linked_lists();              // inseriamo le particelle nelle linked list della griglia
            sort_single_cell_insertionSort(); // ordiniamo le linked list
            check_for_collisions();           // controlliamo le collisioni in tempo lineare
            move_particles();                 // muoviamo le particelle di 1 pixel in una direzione casuale
            reset_linked_lists();             // resettiamo le linked list per il prossimo ciclo

#pragma omp single
            {
                auto end = std::chrono::high_resolution_clock::now(); // calcoliamo la performance
                iteration++;
                if (std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() >= 1000 / config.TARGET_DISPLAY_FPS) {
                    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
                    std::cout << "Iterations per second: " << iteration / (elapsed / 1000.0) << std::endl;
                    start = std::chrono::high_resolution_clock::now();
                    if (config.SHOW_VISUALLY) {
                        // inviamo le particelle via socket all'interfaccia grafica
                        socket_server_send(socket_holder, particles, config.NUM_PARTICLES * sizeof(struct Particle));
                    }
                    iteration = 0;
                }
            }
        }
    }

    return 0;
}
