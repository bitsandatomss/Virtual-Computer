/* sort: quicksort of 150k deterministic ints. Branchy, unpredictable. */
#include <stdio.h>
#define N 400000
static int a[N];
static void swap(int *x, int *y) { int t = *x; *x = *y; *y = t; }
static void qs(int lo, int hi) {
    while (lo < hi) {
        int i = lo, j = hi;
        int pivot = a[lo + (hi - lo) / 2];
        while (i <= j) {
            while (a[i] < pivot) i++;
            while (a[j] > pivot) j--;
            if (i <= j) { swap(&a[i], &a[j]); i++; j--; }
        }
        if (j - lo < hi - i) { if (lo < j) qs(lo, j); lo = i; }
        else { if (i < hi) qs(i, hi); hi = j; }
    }
}
int main(void) {
    unsigned long long s = 987654321ULL;
    for (int i = 0; i < N; i++) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        a[i] = (int)(s >> 33);
    }
    qs(0, N - 1);
    unsigned long long h = 146959ULL;
    for (int i = 0; i < N; i += 3) {
        h ^= (unsigned)(a[i]); h *= 1099511628211ULL;
    }
    printf("%llu %d %d\n", h, a[0], a[N-1]);
    return 0;
}
