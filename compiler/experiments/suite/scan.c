/* scan: prefix sum over 4M doubles. Dependency chain; vectorization-limited. */
#include <stdio.h>
#include <stdlib.h>
#define N 12000000
int main(void) {
    double *a = (double *)malloc(sizeof(double) * N);
    double *p = (double *)malloc(sizeof(double) * N);
    if (!a || !p) return 1;
    for (long i = 0; i < N; i++) a[i] = (double)((i * 2654435761ULL) % 1000) * 0.001;
    double acc = 0;
    for (long i = 0; i < N; i++) { acc += a[i]; p[i] = acc; }
    /* second pass with strided read to defeat clever caching */
    double chk = p[0] + p[N/4] + p[N/2] + p[3*N/4] + p[N-1];
    printf("%.6f\n", chk);
    free(a); free(p);
    return 0;
}
