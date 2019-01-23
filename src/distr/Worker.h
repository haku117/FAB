#pragma once
#include "Runner.h"
#include "IDMapper.h"
#include <atomic>
#include <mutex>

class Worker : public Runner{
public:
	Worker();
	virtual void init(const Option* opt, const size_t lid);
	virtual void run();
	virtual void registerHandlers();
	void bindDataset(const DataHolder* pdh);

private:
	callback_t localCBBinder(void (Worker::*fp)(const std::string&, const RPCInfo&));
	void syncInit();
	void syncProcess();
	void asyncInit();
	void asyncProcess();
	void fsbInit();
	void fsbProcess();
	void fabInit();
	void fabProcess();
	//void generalProcess();

	void updatePointer(const size_t used);
	void sendOnline();
	void waitWorkerList();
	void sendXLength();
	void sendClosed();

	void sendDelta(std::vector<double>& delta);
	void bufferParameter(Parameter& p);
	void applyBufferParameter(); // using the buffer
	void waitParameter();
	void fetchParmeter();

	void pauseTrain();
	void resumeTrain();

// singal
public:
	void handleDelta(const std::string& data, const RPCInfo& info);
	void handleReply(const std::string& data, const RPCInfo& info);
	void handleWorkerList(const std::string& data, const RPCInfo& info);
	void handleParameter(const std::string& data, const RPCInfo& info);
	void handleParameterFsb(const std::string& data, const RPCInfo& info);
	void handleParameterFab(const std::string& data, const RPCInfo& info);
	void handlePause(const std::string& data, const RPCInfo& info);
	void handleContinue(const std::string& data, const RPCInfo& info);
	void handleTerminate(const std::string& data, const RPCInfo& info);
		
	void waitDeltaFromAll();
	void accumulateDelta(std::vector<double>& delta, const int source);
	void applyDelta();
	void sendParameter2M();


private:
	size_t dataPointer;
	size_t iter;
	size_t localBatchSize;
	int ln; // log-every-n times

	int masterNID; // network id of master
	IDMapper wm; // worker mapper
	SyncUnit suOnline;
	SyncUnit suXlength;
	
	bool hasNewParam;
	std::mutex mParam;
	Parameter bfParam;
	SyncUnit suParam;
	//std::mutex mModel; // whether the model is in use

	std::vector<double> bfDelta;

	//std::mutex mTrain;
	std::atomic<bool> allowTrain;
	std::atomic<bool> exitTrain;

	/// new sync for decentrialize
	SyncUnit suDeltaAny, suDeltaAll;
};
