#pragma once
#include "network/RPCInfo.h"
#include "driver/MsgDriver.h"
#include "driver/tools/ReplyHandler.h"
#include "driver/tools/SyncUnit.h"
#include "model/Model.h"
#include "train/GD.h"
#include "common/Statistics.h"
#include <string>
#include <thread>
//#include <chrono>

class NetworkThread;
struct Option;

class Runner{
public:
	const Option* opt;
	size_t nWorker;
	size_t localID; // local logic id
	size_t iter;
	std::string logName;
	Model model;
	GD trainer;
	
public:
	Runner();
	virtual void init(const Option* opt, const size_t lid) = 0;
	virtual void run() = 0;

	void startMsgLoop(const std::string& name = "");
	void stopMsgLoop();
	void msgPausePush();
	void msgPausePop();
	void msgResumePush();
	void msgResumePop();

// helpers:
protected:
	void sleep();
	void sleep(double seconds);
	void msgLoop(const std::string& name = "");

	void finishStat();
	void showStat() const;

// handler helpers
protected:
	using callback_t = std::function<void(const std::string&, const RPCInfo&)>;
	//using callback_t = void (Master::*)(const std::string&, const RPCInfo&);
	//typedef void (Master::*callback_t)(const string&, const RPCInfo&);
	virtual void registerHandlers() = 0;
	void regDSPImmediate(const int type, callback_t fp);
	void regDSPProcess(const int type, callback_t fp);
	void regDSPDefault(callback_t fp);

	void addRPHEach(const int type, std::function<void()> fun, const int n, const bool spawnThread = false);
	void addRPHEachSU(const int type, SyncUnit& su);
	void addRPHAny(const int type, std::function<void()> fun, const bool spawnThread = false);
	void addRPHAnySU(const int type, SyncUnit& su);

	void sendReply(const RPCInfo& info);

	// handlers
public:
	// void handleReply(const std::string& d, const RPCInfo& info);

protected:
	NetworkThread* net;
	std::thread tmsg;

	MsgDriver driver;
	ReplyHandler rph;
	bool msg_do_push;
	bool msg_do_pop;
	bool running;

	Statistics stat;
};
