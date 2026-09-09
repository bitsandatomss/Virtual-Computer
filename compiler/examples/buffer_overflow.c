#include <stdio.h>
#include <string.h>

void safe_function() {
    char buffer[10];
    
    // gets is fundamentally unsafe.
    gets(buffer); 
    
    printf("You entered: %s\n", buffer);
}

int main() {
    safe_function();
    return 0;
}
