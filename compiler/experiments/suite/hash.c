/* hash: FNV-1a over 8M LCG bytes with data-dependent branches. */
#include <stdio.h>
#define N 24000000
int main(void) {
    unsigned long long s = 5555555ULL;
    unsigned long long h = 146959ULL;
    long branches = 0;
    for (long i = 0; i < N; i++) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        unsigned char b = (unsigned char)(s >> 37);
        h ^= b; h *= 1099511628211ULL;
        if ((b & 15) == 0) { h ^= (unsigned long long)i; branches++; }
        else if ((b & 7) == 0) { h += (unsigned long long)b * 31; branches++; }
    }
    printf("%llu %ld\n", h, branches);
    return 0;
}
