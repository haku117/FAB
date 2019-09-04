#pragma once
#include "Runner.h"
#include "IDMapper.h"
#include "util/Timer.h"
#include <atomic>
#include <vector>
#include <list>
#include <mutex>
#include <random>

class Worker : public Runner{
public:
	Worker();
	virtual void init(const Option* opt, const size_t lid);
	virtual void run();
	virtual void registerHandlers();
	void bindDataset(const DataHolder* pdh);

private:
	callback_t localCBBinder(void (Worker::*fp)(std::string&, const RPCInfo&));
	///=== Centralized model
	void syncInit();
	void syncProcess();
	void asyncInit();
	void asyncProcess();
	void fsbInit();
	void fsbProcess();
	void fabInit();
	void fabProcess();
	void progAsyncInit();
	void progAsyncProcess();

	///=== DeCentralized model
	void dcSyncInit();
	void dcSyncProcess();
	void dcFsbProcess();
	void pipeInit();
	void pipeProcess();
	// void dcRingInit();
	// void dcRingProcess();
	// void dcMltInit();
	// void dcMltProcess();
	//void generalProcess();

	///=== initalization functions
	void sendOnline();
	void waitWorkerList();
	void sendXLength();
	void sendClosed();
	void initPipeBlk();

	///=== calculateion functions
	void updatePointer(const size_t used);
	void updatePointerPipe(const size_t used, const size_t blk);

	void pauseTrain() { allowTrain = false; };
	void resumeTrain() { allowTrain = true; };
	void broadcastSignalPause();
	// void broadcastSignalSync() { net->broadcast(MType::CTrainSync, ""); };

	///=== Delta Process functions
	void sendDelta(std::vector<double>& delta, const int ss = -1);
	void sendReport(std::vector<double>& report);
	void broadcastDelta(std::vector<double>& delta);
	void broadcastDeltaPlus(std::vector<double>& delta);
	void ringcastDelta(std::vector<double>& delta);
	void multicastDelta(std::vector<double>& delta);
	void hrkycastDelta(std::vector<double>& delta);
	void grpcastDelta();
	void singlecastDelta(std::vector<double>& delta);
	void dblecastDelta(std::vector<double>& delta);

	void accumulateDelta(const std::vector<double>& delta);
	void accumulateDelta(std::vector<double>& delta, const int source);
	void accumulateDelta(std::vector<double>& delta, const int source, 
		const size_t hlvl, const size_t diter);
	void accumulateDelta(std::vector<double>& delta, const std::vector<int>& sources);
	void accumulateDeltaPipe(std::vector<double>& delta, const int source, const int dIter);
	void copyDelta(std::vector<double>& buffer, std::vector<double>& delta);
	void applyDelta();
	void transmitDelta(int src, int diter);
	void resetDcBuffer();
	void applyDeltaPipe();
	void waitDeltaFromAll();

	///=== Parameter Process functions
	void bufferParameter(Parameter& p);
	void applyBufferParameter(); // using the buffer
	void waitParameter();
	void fetchParmeter();
	void sendParameter2M();

// singal
public:
	void handleDelta(std::string& data, const RPCInfo& info);
	void handleDeltaPipe(std::string& data, const RPCInfo& info);
	void handleDeltaRingcast(std::string& data, const RPCInfo& info);
	void handleDeltaMltcast(std::string& data, const RPCInfo& info);
	void handleDeltaHrkycast(std::string& data, const RPCInfo& info);
	void handleDeltaGrpcast(std::string& data, const RPCInfo& info);
	void handleDeltaRPL(std::string& data, const RPCInfo& info);
	void handleDeltaRPLone(std::string& data, const RPCInfo& info);
	void handleDeltaRPLtrans(std::string& data, const RPCInfo& info);
	void handleDelta2c(std::string& data, const RPCInfo& info);
	void handleDeltaRequest(std::string& data, const RPCInfo& info);

	void handleReply(std::string& data, const RPCInfo& info);
	void handleWorkerList(std::string& data, const RPCInfo& info);
	void handleParameter(std::string& data, const RPCInfo& info);
	void handleParameterFab(std::string& data, const RPCInfo& info);
	void handleParameterFsb(std::string& data, const RPCInfo& info);
	void handleParameterAsync(std::string& data, const RPCInfo& info);
	void handleParameterProg(std::string& data, const RPCInfo& info);

	void handlePause(std::string& data, const RPCInfo& info);
	void handleSync(std::string& data, const RPCInfo& info);
	void handleContinue(std::string& data, const RPCInfo& info);
	void handleTerminate(std::string& data, const RPCInfo& info);
	void handleInterval(std::string& data, const RPCInfo& info);
		
// util
	size_t id2lvl(const size_t id);
	int dstGrpID(const size_t id, const size_t lvl);

private:
	size_t dataPointer;
	size_t localBatchSize;
	size_t reportSize;
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
	int curCnt;
	double forceCalT;
	int mltDD;
	size_t curHlvl;
	size_t mylvl;
	size_t dstgrpID;
	double interval;

	// int interval;
	// int delayWorkers;
	std::mt19937 gen;
	std::exponential_distribution<double> distribution;
	double lamda;
	size_t range;
	std::vector<double> delayArr;
	size_t paramVersion;

	int blkNum, nny; // block size for pipeline running
	int stale; // record the current iter for updated param
	int staleness; // max staleness allowed to update param
	std::vector<int> blkPointer; // block start and index??
	std::vector<int> blkSize; // block start and index??
	std::vector<int> blkDeltaBFCnt; // count the delta buffered at each blk
	std::vector<std::vector<double> > bfBlkDelta;
	std::vector<std::vector<bool> > bfBlkDeltaIndex;
	int curDeltaIndex; // indicate the position of current delta in bf

	std::vector<double> bufferDelta;	// buffer the delta from other workers
	std::vector<double> bufferDeltaExt;	// buffer the multiple delta from other workers
	std::vector<int> accuDelta; // accumulate delta stats
	std::vector<int> accuDeltaExt;
	std::vector<double> bufferDeltaLeftover;	// buffer the delta from other workers
	std::vector<double> bufferDeltaLeftoverExt;	// buffer the multiple delta from other workers
	std::vector<bool> deltaIndx0;		// delta buffer indicator
	std::vector<bool> deltaIndx1;
	std::vector<bool> deltaReceiver;
	std::vector<double> deltaWaitT;	// record the delta arriving time
	int bfDeltaCnt, bfDeltaCntExt;
	bool isbfDeltaExt;
	std::vector<int> recSrcs; // list to send deltaRPL
	
	std::mutex mParam;
	std::mutex mDelta;
	Parameter bfParam;
	SyncUnit suParam;
	bool hasNewParam;
	bool reqDelta;
	//std::mutex mModel; // whether the model is in use

	std::vector<double> bfDelta; // dcsync need to delete
	std::list< std::vector<double> > staleDelta; // dcsync need to delete

	//std::mutex mTrain;
	std::atomic<bool> allowTrain;
	std::atomic<bool> exitTrain;

	/// new sync for decentrialize
	SyncUnit suDeltaAny, suDeltaAll;
	// SyncUnit suTPause; // for dc fsb check pause status
};
