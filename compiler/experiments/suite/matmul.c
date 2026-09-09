/* matmul: dense N=384 double GEMM, ijk order. Compute-bound, vectorizer target. */
#include <stdio.h>
#define N 768
static double A[N][N], B[N][N], C[N][N];
int main(void) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            A[i][j] = (i * 31 + j * 17) % 97 + 0.5;
            B[i][j] = (i * 13 - j * 7) % 89 + 0.25;
            C[i][j] = 0.0;
        }
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            double s = 0.0;
            for (int k = 0; k < N; k++) s += A[i][k] * B[k][j];
            C[i][j] = s;
        }
    unsigned long long h = 146959ULL;
    for (int i = 0; i < N; i += 7)
        for (int j = 0; j < N; j += 7) {
            union { double d; unsigned long long u; } v = {C[i][j]};
            h ^= v.u; h *= 1099511628211ULL;
        }
    printf("%llu\n", h);
    return 0;
}
