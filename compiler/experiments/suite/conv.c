/* conv: 1M-sample 1D convolution, 64 taps. Streaming + vectorizer target. */
#include <stdio.h>
#define N 3000000
#define TAPS 64
static float x[N + TAPS], c[TAPS], y[N];
int main(void) {
    for (int i = 0; i < N + TAPS; i++)
        x[i] = (float)(((i * 1103515245ULL + 12345) >> 16) % 2000) * 0.001f - 1.0f;
    for (int t = 0; t < TAPS; t++) c[t] = 1.0f / (t + 1);
    for (int i = 0; i < N; i++) {
        float s = 0;
        for (int t = 0; t < TAPS; t++) s += x[i + t] * c[t];
        y[i] = s;
    }
    double chk = 0;
    for (int i = 0; i < N; i += 97) chk += y[i];
    printf("%.6f\n", chk);
    return 0;
}
