#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

// Function to find the optimal max_depth
int find_optimal_max_depth() {
    int num_procs = omp_get_num_procs(); // Number of processors available
    int max_depth = 0;

    // Heuristic: set max_depth such that 2^max_depth is close to num_procs
    while ((1 << max_depth) < num_procs) {
        max_depth++;
    }
    printf("Optimal max_depth: %d\n", max_depth);
    return max_depth;
}

// Function to merge two halves of an array
void merge(int arr[], int l, int m, int r) {
    int i, j, k;
    int n1 = m - l + 1;
    int n2 = r - m;

    // Temporary arrays
    int *L = (int *)malloc(n1 * sizeof(int));
    int *R = (int *)malloc(n2 * sizeof(int));

    for (i = 0; i < n1; i++) {
        L[i] = arr[l + i];
    }
    for (j = 0; j < n2; j++) {
        R[j] = arr[m + 1 + j];
    }

    i = 0;
    j = 0;
    k = l;

    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k++] = L[i++];
        } else {
            arr[k++] = R[j++];
        }
    }

    while (i < n1) {
        arr[k++] = L[i++];
    }

    while (j < n2) {
        arr[k++] = R[j++];
    }

    free(L);
    free(R);
}

// Recursive merge sort function
void merge_sort(int arr[], int l, int r, int depth) {
    if (l < r) {
        int m = l + (r - l) / 2;

        if (depth > 0) {
#pragma omp task shared(arr) firstprivate(l, m, r, depth)
            {
                // printf("Thread %d is sorting left part from %d to %d\n", omp_get_thread_num(), l, m);
                merge_sort(arr, l, m, depth - 1);
            }

#pragma omp task shared(arr) firstprivate(l, m, r, depth)
            {
                // printf("Thread %d is sorting right part from %d to %d\n", omp_get_thread_num(), m + 1, r);
                merge_sort(arr, m + 1, r, depth - 1);
            }

#pragma omp taskwait
        } else {
            merge_sort(arr, l, m, 0);
            merge_sort(arr, m + 1, r, 0);
        }

        // printf("Thread %d is merging parts from %d to %d\n", omp_get_thread_num(), l, r);
        merge(arr, l, m, r);
    }
}

// Parallel merge sort function
void parallel_merge_sort(int arr[], int size) {
    int max_depth = find_optimal_max_depth(); // You can adjust this value based on experimentation
#pragma omp parallel
    {
#pragma omp single
        {
            merge_sort(arr, 0, size - 1, max_depth);
        }
    }
}

// Helper function to print the array
void printArray(int arr[], int size) {
    for (int i = 0; i < size; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

int main() {
    int size = 1000000000;
    // create random array
    int *arr = (int *)malloc(size * sizeof(int));
    for (int i = 0; i < size; i++) {
        arr[i] = rand() % size;
    }

    double start, end;

    // // Measure performance of sequential merge sort
    // int *seq_arr = (int *)malloc(size * sizeof(int));
    // for (int i = 0; i < size; i++) {
    //     seq_arr[i] = arr[i];
    // }
    // start = omp_get_wtime();
    // merge_sort(seq_arr, 0, size - 1, 0); // Depth 0 for sequential version
    // end = omp_get_wtime();
    // printf("Sequential merge sort time: %f\n", end - start);

    // Measure performance of parallel merge sort
    int *par_arr = (int *)malloc(size * sizeof(int));
    for (int i = 0; i < size; i++) {
        par_arr[i] = arr[i];
    }
    start = omp_get_wtime();
    parallel_merge_sort(par_arr, size);
    end = omp_get_wtime();
    printf("Parallel merge sort time: %f\n", end - start);

    // Print sorted arrays for verification
    // printf("Sequentially sorted array:\n");
    // printArray(seq_arr, size);

    // printf("Parallel sorted array:\n");
    // printArray(par_arr, size);

    // free(seq_arr);
    free(par_arr);
    free(arr);

    return 0;
}
