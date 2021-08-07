#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int
main(int argc, char *argv[]){
#pragma omp parallel
	{
		printf("hello world\n");
	}
	return 0;
}

/* コンパイルと実行
gcc -fopenmp ファイル名.c
./a.out
スレッドの指定方法
・プログラム内：omp_set_num_threads(コア数);
・実行時に指定：env OMP_NUM_THREADS=16 ./a.out
*/
