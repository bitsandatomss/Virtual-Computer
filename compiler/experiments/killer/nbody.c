/* nbody: 1500 bodies, 10 leapfrog steps, float. Compute-bound, sqrt-heavy. */
#include <stdio.h>
#include <math.h>
#define NB 3000
#define STEPS 12
static float px[NB], py[NB], pz[NB], vx[NB], vy[NB], vz[NB], m[NB];
int main(void) {
    unsigned long long s = 12345ULL;
    for (int i = 0; i < NB; i++) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        px[i] = (float)((s >> 11) % 1000) * 0.01f;
        py[i] = (float)((s >> 21) % 1000) * 0.01f;
        pz[i] = (float)((s >> 31) % 1000) * 0.01f;
        vx[i] = vy[i] = vz[i] = 0.0f;
        m[i] = 1.0f + (float)(i % 10) * 0.1f;
    }
    for (int t = 0; t < STEPS; t++) {
        for (int i = 0; i < NB; i++) {
            float fx = 0, fy = 0, fz = 0;
            for (int j = 0; j < NB; j++) {
                if (i == j) continue;
                float dx = px[j] - px[i], dy = py[j] - py[i], dz = pz[j] - pz[i];
                float r2 = dx*dx + dy*dy + dz*dz + 0.01f;
                float inv = 1.0f / sqrtf(r2 * r2 * r2);
                fx += dx * inv * m[j]; fy += dy * inv * m[j]; fz += dz * inv * m[j];
            }
            vx[i] += fx * 0.001f; vy[i] += fy * 0.001f; vz[i] += fz * 0.001f;
        }
        for (int i = 0; i < NB; i++) {
            px[i] += vx[i]; py[i] += vy[i]; pz[i] += vz[i];
        }
    }
    double chk = 0;
    for (int i = 0; i < NB; i++) chk += px[i] + py[i] * 2 + pz[i] * 3;
    printf("%.6f\n", chk);
    return 0;
}
