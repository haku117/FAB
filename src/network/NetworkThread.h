#pragma once
#include <thread>
#include <mutex>
#include <functional>
#include <string>
#include <deque>
#include <vector>
#include "Task.h"
#include "serial/serialization.h"
#include "util/Timer.h"

class NetworkImplMPI;

// Hackery to get around mpi's unhappiness with threads.  This thread
// simply polls MPI continuously for any kind of update and adds it to
// a local queue.
class NetworkThread{
public:
	bool active() const;
	size_t pending_pkgs() const;
	int64_t pending_bytes() const;
	size_t unpicked_pkgs() const{
		return receive_buffer.size();
	}
	int64_t unpicked_bytes() const;
	
	// Blocking read for the given source and message type.
	void readAny(std::string& data, int *sourcsrcRete=nullptr, int *typeRet=nullptr);
	// Unblocked read for the given source and message type.
	bool tryReadAny(std::string& data, int *sosrcReturce=nullptr, int *typeRet=nullptr);

	// Enqueue the given request to pending buffer for transmission.
	template <class T>
	void send(int dst, int tag, const T& msg) {
		Timer tmr;
		std::string s = serialize(msg);
		stat_time_serial += tmr.elapseSd();
		send(new Task(dst, tag, move(s)));
		stat_send_time += tmr.elapseSd();
	}
	void send(int dst, int tag, std::string&& msg) {
		Timer tmr;
		send(new Task(dst, tag, move(msg)));
		stat_send_time += tmr.elapseSd();
	}

	template <class T>
	void broadcast(int tag, const T& msg) {
		Timer tmr;
		std::string s = serialize(msg);
		stat_time_serial += tmr.elapseSd();
		broadcast(new Task(Task::ANY_DST, tag, move(s)));
		stat_send_time += tmr.elapseSd();
	}

	void flush();

	int id() const;
	int size() const;

	static NetworkThread *GetInstance();
	static void Init(int argc, char* argv[]);
	// finish current tasks and terminate
	static void Shutdown();
	// abandon ongoing tasks and terminate all network related functions
	static void Terminate();

	bool pause_=false;

	uint64_t stat_send_pkg, stat_recv_pkg;
	uint64_t stat_send_byte, stat_recv_byte;
	double stat_send_time, stat_recv_time;
	double stat_time_serial;

private:
	bool running;
	bool done;
	NetworkImplMPI* net;
	mutable std::thread t_;

	//buffer for request to be sent, double buffer design for performance
	std::vector<std::pair<Task*, bool>> ps_buffer_[2]; // pair(task, broadcast)
	std::vector<std::pair<Task*, bool>>* pending_sends_;
	unsigned ps_idx_=0;
	mutable std::recursive_mutex ps_lock;

	std::deque<std::pair<std::string,TaskBase> > receive_buffer;
	mutable std::recursive_mutex rec_lock;

	// Enqueue the given request to pending buffer for transmission.
	void send(Task *req);
	void broadcast(Task *req);


	bool checkReceiveQueue(std::string& data, TaskBase& info);

	void Run();

	static NetworkThread* self;
	NetworkThread();
};

