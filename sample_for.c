
int
main(int argc, char *argv[]){
#pragma omp parallel
	{
		printf("hello world\n");
	}
	return 0;
}
