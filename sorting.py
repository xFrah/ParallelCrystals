import random

def move(arr2):
    # add +1 or -1 to the elements in arr2, store the +1 or -1 in arr1
    arr1 = [random.choice([-1, 1]) for _ in range(len(arr2))]
    for i in range(len(arr1)):
        arr2[i] += arr1[i]
    print(arr2)
    return arr1, arr2


if __name__ == "__main__":
    arr1, arr2 = move(sorted([random.randint(2, 100) for i in range(30)]))
    print(arr1)
    print(arr2)

    for i in range(1, len(arr1)):
        if arr1[i] == 1:
            print("Skipping")
            continue
        for j in range(i, 0, -1):
            print(f"Going back {j}")
            if j == 0:
                print("Reached 0")
                break
            if arr1[j - 1] == 1:
                print(f"Trying to swap {j} and {j-1}")
                if arr2[j] < arr2[j - 1]:
                    print(f"Swapping {arr2[j - 1]} ({j - 1}) and {arr2[j]} ({j})")
                    arr2[j], arr2[j - 1] = arr2[j - 1], arr2[j]
                    arr1[j], arr1[j - 1] = arr1[j - 1], arr1[j]
                    print(arr2)
                else:
                    print(f"Not swapping {arr2[j-1]} ({j - 1}) and {arr2[j]} ({j})")
                    break
    print(arr2)
