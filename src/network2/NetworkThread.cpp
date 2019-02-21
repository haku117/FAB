#include "NetworkThread.h"
#include "NetworkImplMPI.h"
#include "logging/logging.h"
#include <chrono>
using namespace std;

double FLAGS_sleep_time = 0.001;

static inline void Sleep(){
	this_thread::sleep_for(chrono::duration<double>(FLAGS_sleep_time));
}

NetworkThread::NetworkThread() :
		stat_send_pkg(0), stat_recv_pkg(0),
		stat_send_byte(0), stat_recv_byte(0),
		net(nullptr)
{
	net = NetworkImplMPI::GetInstance();
}

int NetworkThread::id() const{
	return net->id();
}
int NetworkThread::size() const{
	return net->size();
}

void NetworkThread::readAny(string& data, int *srcRet, int *typeRet){
//	Timer t;
	while(!tryReadAny(data, srcRet, typeRet)){
		Sleep();
	}
//	stats["network_time"] += t.elapsed();
}
bool NetworkThread::tryReadAny(string& data, int *srcRet, int *typeRet){
	TaskHeader hdr;
	if(net->probe(&hdr)){
		string rd = net->receive(&hdr);
		DVLOG(3) << "RP from " << hdr.src_dst << " type " << hdr.type;
		data = move(rd);
		if(srcRet) *srcRet = hdr.src_dst;
		if(typeRet) *typeRet = hdr.type;
		return true;
	}
	return false;
}

// Enqueue the given request to pending buffer for transmission.
int NetworkThread::send(Task *req){
	int size = req->payload.size();
	stat_send_byte += size;
	stat_send_pkg++;
	DVLOG(3) << "SP to " << req->src_dst << " type " << req->type;
	net->send(req);
	return size;
}

int NetworkThread::broadcast(Task* req) {
	int size = req->payload.size();
	net->broadcast(req);
	++stat_send_pkg;
	stat_send_byte += size;
	return size;
}


// ---------------- singleton related ------------------

NetworkThread* NetworkThread::self = nullptr;

NetworkThread* NetworkThread::GetInstance(){
	return self;
}

void NetworkThread::Init(int argc, char* argv[]){
	NetworkImplMPI::Init(argc, argv);
	self = new NetworkThread();
	atexit(&NetworkThread::Terminate);
}

void NetworkThread::Shutdown() {
	// LOG(INFO) << " Shutdown thread net2 ";
	if(self != nullptr) {
		NetworkThread* p = nullptr;
		swap(self, p); // use the swap primitive to preform safe deletion
		delete self;
		NetworkImplMPI::Shutdown();
		// LOG(INFO) << " Shutdown MPI done ";
	}
}

void NetworkThread::Terminate()
{
	Shutdown();
}
