/* stencil: 512x512 double 5-point heat, 40 iterations. Memory-ish bound. */
#include <stdio.h>
#define W 768
#define H 768
#define IT 64
static double A[H][W], B[H][W];
int main(void) {
    for (int i = 0; i < H; i++)
        for (int j = 0; j < W; j++)
            A[i][j] = ((i * 131 + j * 17) % 100) * 0.01;
    for (int t = 0; t < IT; t++) {
        for (int i = 1; i < H - 1; i++)
            for (int j = 1; j < W - 1; j++)
                B[i][j] = 0.25 * (A[i-1][j] + A[i+1][j] + A[i][j-1] + A[i][j+1]);
        for (int i = 1; i < H - 1; i++)
            for (int j = 1; j < W - 1; j++)
                A[i][j] = B[i][j];
    }
    double chk = 0;
    for (int i = 0; i < H; i += 5)
        for (int j = 0; j < W; j += 5) chk += A[i][j];
    printf("%.6f\n", chk);
    return 0;
}
