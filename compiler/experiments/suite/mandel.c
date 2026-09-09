/* mandel: 768x512 Mandelbrot, maxiter 256. Compute-bound, divergent branches. */
#include <stdio.h>
#define W 1024
#define H 768
#define MAXIT 256
int main(void) {
    long inside = 0;
    long hist[8] = {0,0,0,0,0,0,0,0};
    for (int py = 0; py < H; py++) {
        double ci = (py - H / 2) * 4.0 / W;
        for (int px = 0; px < W; px++) {
            double cr = (px - W / 2) * 4.0 / W;
            double zr = 0, zi = 0;
            int it = 0;
            while (it < MAXIT && zr * zr + zi * zi < 4.0) {
                double t = zr * zr - zi * zi + cr;
                zi = 2 * zr * zi + ci;
                zr = t;
                it++;
            }
            if (it == MAXIT) inside++;
            hist[it & 7] += it;
        }
    }
    printf("%ld %ld %ld %ld %ld %ld %ld %ld %ld\n", inside,
           hist[0], hist[1], hist[2], hist[3], hist[4], hist[5], hist[6],
           hist[7]);
    return 0;
}
