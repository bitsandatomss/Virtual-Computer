#include <stdio.h>
#include <string.h>

static unsigned long checksum(const unsigned char *buf, unsigned long n) {
    unsigned long h = 146959u;
    for (unsigned long i = 0; i < n; i++) {
        h ^= buf[i];
        h *= 1099511628211u;
        h ^= h >> 13;
    }
    return h;
}

int main(void) {
    static unsigned char buf[8192];
    for (unsigned long i = 0; i < sizeof(buf); i++) {
        buf[i] = (unsigned char)((i * 31u + 7u) & 0xffu);
    }
    volatile unsigned long acc = 0;
    for (int r = 0; r < 500; r++) {
        acc += checksum(buf, sizeof(buf));
        buf[r % sizeof(buf)] ^= (unsigned char)r;
    }
    printf("%lu\n", acc);
    return 0;
}
