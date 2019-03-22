#pragma once
#include "Runner.h"
#include "IDMapper.h"
#include "util/Timer.h"
#include <atomic>
#include <vector>
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
	void dcSyncInit();
	void dcSyncProcess();
	void dcFsbProcess();
	void dcRingInit();
	void dcRingProcess();
	//void generalProcess();

	void updatePointer(const size_t used);
	void sendOnline();
	void waitWorkerList();
	void sendXLength();
	void sendClosed();

	void accumulateDelta(const std::vector<double>& delta);
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
	void handleDeltaRingcast(const std::string& data, const RPCInfo& info);
	void handleReply(const std::string& data, const RPCInfo& info);
	void handleWorkerList(const std::string& data, const RPCInfo& info);
	void handleParameter(const std::string& data, const RPCInfo& info);
	void handleParameterFab(const std::string& data, const RPCInfo& info);
	void handleParameterFsb(const std::string& data, const RPCInfo& info);
	void handlePause(const std::string& data, const RPCInfo& info);
	void handleContinue(const std::string& data, const RPCInfo& info);
	void handleTerminate(const std::string& data, const RPCInfo& info);
		
	void broadcastDelta(std::vector<double>& delta);
	void ringcastDelta(std::vector<double>& delta);
	void multicastDelta(std::vector<double>& delta);
	void waitDeltaFromAll();
	void accumulateDelta(std::vector<double>& delta, const int source);
	void copyDelta(std::vector<double>& buffer, std::vector<double>& delta);
	void applyDelta();
	void sendParameter2M();
	void broadcastSignalPause();

private:
	size_t dataPointer;
	size_t localBatchSize;
	int ln; // log-every-n times

	int masterNID; // network id of master
	IDMapper wm; // worker mapper
	SyncUnit suOnline;
	SyncUnit suXlength;

	int typeDDeltaAny, typeDDeltaAll;
	//workerLst = {}; // ??
	//trainer.bindModel(&model);
	double factorDelta;
	size_t nx;
	//iter = 0;
	size_t nUpdate;
	size_t lastArchIter;		
	Timer tmrGlb; // for monitoring the delta ariving time
	double curCalT;

	std::vector<double> bufferDelta;	// buffer the delta from other workers
	std::vector<double> bufferDeltaExt;	// buffer the multiple delta from other workers
	std::vector<bool> deltaIndx0;		// delta buffer indicator
	std::vector<bool> deltaIndx1;
	std::vector<bool> deltaReceiver;
	std::vector<double> deltaWaitT;	// record the delta arriving time
	int bfDeltaCnt;
	
	bool hasNewParam;
	std::mutex mParam;
	std::mutex mDelta;
	Parameter bfParam;
	SyncUnit suParam;
	//std::mutex mModel; // whether the model is in use

	std::vector<double> bfDelta; // dcsync need to delete

	//std::mutex mTrain;
	std::atomic<bool> allowTrain;
	std::atomic<bool> exitTrain;

	/// new sync for decentrialize
	SyncUnit suDeltaAny, suDeltaAll;
	// SyncUnit suTPause; // for dc fsb check pause status
};
