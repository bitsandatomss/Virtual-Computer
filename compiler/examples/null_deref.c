#include <stddef.h>

static int read_value(const int *value) {
    return *value;
}

static int compute(void) {
    return read_value(NULL);
}

int main(void) {
    return compute();
}
