#include <stdio.h>

#define N 96

static double A[N][N], B[N][N], C[N][N];

int main(void) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = (double)(i + j) * 0.5;
            B[i][j] = (double)(i - j) * 0.25;
            C[i][j] = 0.0;
        }
    }
    for (int r = 0; r < 12; r++) {
        for (int i = 0; i < N; i++) {
            for (int k = 0; k < N; k++) {
                double aik = A[i][k];
                for (int j = 0; j < N; j++) {
                    C[i][j] += aik * B[k][j];
                }
            }
        }
    }
    printf("%.2f\n", C[0][0] + C[N - 1][N - 1]);
    return 0;
}
