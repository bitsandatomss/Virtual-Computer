#include <stdio.h>

static unsigned long mix(unsigned long x) {
    if ((x & 1u) == 0u) {
        x >>= 1;
    } else {
        x = x * 3u + 1u;
    }
    switch (x % 7u) {
        case 0:
            x += 11u;
            break;
        case 3:
            x ^= 0x9e3779b9u;
            break;
        default:
            x += (x % 13u);
            break;
    }
    return x;
}

int main(void) {
    volatile unsigned long acc = 0;
    for (unsigned long r = 0; r < 400000u; r++) {
        acc += mix(r);
    }
    printf("%lu\n", acc);
    return 0;
}
