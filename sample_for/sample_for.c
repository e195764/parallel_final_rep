#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <omp.h>
 
int
main(int argc, char *argv[])
{
    size_t i;
#pragma omp parallel for
    for(i = 0; i < 10; ++i)
    {
        sleep (1);
        printf("hello world: %lu\n", i);
    }
    exit(EXIT_SUCCESS);
}
