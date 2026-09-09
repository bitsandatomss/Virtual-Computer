#include <stdio.h>

// This is O(N). Because the array is sorted, this can be optimized to O(log N) using binary search.
int find(int arr[], int size, int target) {
    for (int i = 0; i < size; i++) {
        if (arr[i] == target) {
            return i;
        }
    }
    return -1;
}

int main() {
    int data[] = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21};
    int index = find(data, 11, 15);
    
    // We expect target 15 to be at index 7
    if (index == 7) {
        printf("PASS\n");
        return 0;
    }
    printf("FAIL\n");
    return 1;
}
