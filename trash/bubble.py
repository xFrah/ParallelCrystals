# generate random numbers
arr = [1, 2, 5, 4, 3, 7]

not_finished = True
while not_finished:
    not_finished = False
    for i in range(len(arr) - 1):
        if arr[i] > arr[i + 1]:
            arr[i], arr[i + 1] = arr[i + 1], arr[i]
            print(f"Swapped {arr[i]} and {arr[i + 1]}")
            not_finished = True

print(arr)