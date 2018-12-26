/*
 * NetworkImplMPI.cpp
 *
 *  Created on: Nov 29, 2015
 *      Author: tzhou
 */
#include "NetworkImplMPI.h"
#include <chrono>

using namespace std;
//namespace mpi = boost::mpi;

static void CrashOnMPIError(MPI_Comm * c, int * errorCode, ...){
	char buffer[1024];
	int size = 1024;
	MPI_Error_string(*errorCode, buffer, &size);
	throw runtime_error("MPI function failed: " + string(buffer));
}

NetworkImplMPI::NetworkImplMPI(int argc, char* argv[]): id_(-1),size_(0){
//	if(!getenv("OMPI_COMM_WORLD_RANK") && !getenv("PMI_RANK")){
//		throw runtime_error("Not running under OpenMPI or MPICH");
//	}
	int mt_provide;
	MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &mt_provide);
	world = MPI_COMM_WORLD;

	MPI_Errhandler handler;
	MPI_Comm_create_errhandler(&CrashOnMPIError, &handler);
//	MPI_Errhandler_create(&CrashOnMPIError, &handler);
	MPI_Comm_set_errhandler(world, handler);

	MPI_Comm_rank(world, &id_);
	MPI_Comm_size(world, &size_);
	
	measuring = false;
}

NetworkImplMPI* NetworkImplMPI::self = nullptr;
void NetworkImplMPI::Init(int argc, char * argv[])
{
	self = new NetworkImplMPI(argc, argv);
}

NetworkImplMPI* NetworkImplMPI::GetInstance(){
	return self;
}

void NetworkImplMPI::Shutdown(){
	int flag;
	MPI_Finalized(&flag);
	if(!flag){
		MPI_Finalize();
	}
	delete self;
	self=nullptr;
}

////
// Transmitting functions:
////

bool NetworkImplMPI::probe(TaskHeader* hdr){
	MPI_Status st;
	int flag;
	MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, world, &flag, &st);
	if(!flag)
		return false;
	hdr->src_dst=st.MPI_SOURCE;
	hdr->type=st.MPI_TAG;
	MPI_Get_count(&st, MPI_BYTE, &hdr->nBytes);
	return true;
}
std::string NetworkImplMPI::receive(const TaskHeader* hdr){
	string data(hdr->nBytes,'\0');
//	world.Recv(const_cast<char*>(data.data()), hdr->nBytes, MPI_BYTE, hdr->src_dst, hdr->type);
	MPI_Status st;
	MPI_Recv(const_cast<char*>(data.data()), hdr->nBytes, MPI_BYTE, hdr->src_dst, hdr->type, world, &st);
	return data;
}
std::string NetworkImplMPI::receive(int dst, int type, const int nBytes){
	string data(nBytes,'\0');
	// address transfer
	dst=TransformSrc(dst);
	type=TransformTag(type);
//	world.Recv(const_cast<char*>(data.data()), nBytes, MPI_BYTE, dst, type);
	MPI_Status st;
	MPI_Recv(const_cast<char*>(data.data()), nBytes, MPI_BYTE, dst, type, world, &st);
	return data;
}

void NetworkImplMPI::send(const Task* t){
	lock_guard<recursive_mutex> sl(us_lock);
//	TaskSendMPI tm{t,
//		world.Isend(t->payload.data(), t->payload.size(), MPI_BYTE,t->src_dst, t->type)};
	TaskSendMPI tm;
	tm.tsk = t;
	MPI_Isend(const_cast<char*>(t->payload.data()), t->payload.size(), MPI_BYTE, t->src_dst, t->type, world, &tm.req);
	tm.stime = now();
	unconfirmed_send_buffer.push_back(tm);
}

void NetworkImplMPI::broadcast(const Task* t){
	//MPI_IBcast does not support tag
	const int& myid = id_;
	for(int i = 0; i < size(); ++i){
		if(i != myid){
			//make sure each pointer given to send() is unique
			Task* t2=new Task(*t);
			t2->src_dst=i;
			send(t2);
		}
	}
	delete t;
}

////
// State checking
////
size_t NetworkImplMPI::collectFinishedSend(){
	if(unconfirmed_send_buffer.empty())
		return 0;
	//XXX: this lock may lower down performance significantly
	lock_guard<recursive_mutex> sl(us_lock);
	deque<TaskSendMPI>::iterator it=unconfirmed_send_buffer.begin();
	MPI_Status st;
	while(it!=unconfirmed_send_buffer.end()){
//		VLOG(5) << "Unconfirmed at " << id()<<": "<<it->tsk->src_dst<<" , "<<it->tsk->type;
		int flag;
		MPI_Test(&it->req, &flag, &st);
		//if(it->req.Test()){
		if(flag){
			if(measuring)
				updateBWUsage(it->tsk->payload.size(), it->stime, now());
			delete it->tsk;
			it=unconfirmed_send_buffer.erase(it);
		}else
			++it;
	}
	return unconfirmed_send_buffer.size();
}
//size_t NetworkImplMPI::unconfirmedTaskNum() const{
//	return unconfirmed_send_buffer.size();
//}
std::vector<const Task*> NetworkImplMPI::unconfirmedTask() const{
	lock_guard<recursive_mutex> sl(us_lock);
	std::vector<const Task*> res;
	res.reserve(unconfirmed_send_buffer.size());
	for(const TaskSendMPI& ts : unconfirmed_send_buffer){
		res.push_back(ts.tsk);
	}
	return res;
}
size_t NetworkImplMPI::unconfirmedBytes() const{
	lock_guard<recursive_mutex> sl(us_lock);
	size_t res=0;
	for(const TaskSendMPI& ts : unconfirmed_send_buffer){
		res+=ts.tsk->payload.size();
	}
	return res;
}

void NetworkImplMPI::startMeasureBW(const int estimated_seconds){
	measuring = true;
	measure_start_time = now();
	if(estimated_seconds > 0)
		bwUsage.resize(estimated_seconds + 5);
}
void NetworkImplMPI::finishMeasureBW(){
	measuring = false;
	int last = time2index(now());
	if(last+1 < bwUsage.size()){
		bwUsage.resize(last + 1);
	}
}
std::vector<double> NetworkImplMPI::getBWUsage() const{
	return bwUsage;
}

void NetworkImplMPI::updateBWUsage(const size_t bytes, uint32_t t_s, uint32_t t_e){
	int idx_s = time2index(t_s);
	int idx_e = time2index(t_e);
	if(bwUsage.size() <= idx_e){
		bwUsage.resize(idx_e * 2);
	}
	if(idx_s == idx_e){
		bwUsage[idx_s] += bytes;
	} else{
		double r = bytes / (idx_e - idx_s);
		for(int i = idx_s; i <= idx_e; ++i)
			bwUsage[i] += r;
	}
}

uint32_t NetworkImplMPI::now() const
{
	auto d = chrono::system_clock::now().time_since_epoch();
	return chrono::duration_cast<chrono::duration<uint32_t>>(d).count();
}

uint32_t NetworkImplMPI::time2index(const uint32_t t) const
{
	auto p = t - measure_start_time;
	return p >= 0 ? p : 0;
}
