#include <iostream>
#include <mpi.h>

using namespace std;

int main(int argc, char *argv[]){
	int mpi_mt;
	MPI_Init_thread(&argc,&argv,MPI_THREAD_MULTIPLE,&mpi_mt);

	int rank, size;
	char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Get_processor_name(processor_name, &name_len);
	if(rank == 0){
		cout<<"MPI multiple thread support level: "<<mpi_mt<<"\t";
		cout<<"(MULTIPLE : "<<MPI_THREAD_MULTIPLE<<" ; SERIALIZED : "<<MPI_THREAD_SERIALIZED
			<<" ; FUNNELED : "<<MPI_THREAD_FUNNELED<<" ; SINGLE : "<<MPI_THREAD_SINGLE<<")"<<endl;
	}
	MPI_Barrier(MPI_COMM_WORLD);
	cout<<"size: "<<size<<", rank: "<<rank<<", processor: "<<processor_name<<endl;

	MPI_Finalize();
	return 0;
}
