#include <stdio.h>

static long linear_search(const long *a, long n, long key) {
    long found = -1;
    for (long i = 0; i < n; i++) {
        if (a[i] == key) {
            found = i;
        }
    }
    return found;
}

int main(void) {
    long a[2048];
    for (long i = 0; i < 2048; i++) {
        a[i] = i * 3 + 1;
    }
    volatile long acc = 0;
    for (long r = 0; r < 300; r++) {
        acc += linear_search(a, 2048, 6142);
    }
    printf("%ld\n", acc);
    return 0;
}
