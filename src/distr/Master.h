#pragma once
#include "Runner.h"
#include "IDMapper.h"
#include "IntervalEstimator.h"
#include "driver/tools/SyncUnit.h"
#include "util/Timer.h"
#include "train/EM.h"
#include "train/GD.h"
#include <vector>
#include <fstream>

class Master : public Runner{
public:
	Master();
	virtual void init(const Option* opt, const size_t lid);
	virtual void run();
	virtual void registerHandlers();
	void bindDataset(const DataHolder* pdh);

private:
	callback_t localCBBinder(void (Master::*fp)(const std::string&, const RPCInfo&));
	void syncInit();
	void syncProcess();
	void asyncInit();
	void asyncProcess();
	void fsbInit();
	void fsbProcess();
	void fabInit();
	void fabProcess();
	void dcInit();
	void dcProcess();
	void progAsyncInit();
	void progAsyncProcess();

// local logic
private:
	void applyDelta(std::vector<double>& delta, const int source);
	//void receiveDelta(std::vector<double>& delta, const int source);
	bool terminateCheck();
	void initializeParameter();
	void sendParameter(const int target, const bool sendVersion = false);
	void broadcastParameter(const bool sendVersion = false);
	void waitParameterConfirmed();
	bool needArchive();
	bool needArchiveAsync(int it);
	bool archiveProgress(const bool force = false);
	bool archiveProgressAsync(std::string staleStats = "", const bool force = false);

// signal logic
public:
	void broadcastWorkerList();
	void broadcastSignalPause();
	void broadcastSignalContinue();
	void broadcastSignalTerminate();
	void waitDeltaFromAny(); // dont reset suDeltaAny
	void waitDeltaFromAll(); // reset suDeltaAll
	void gatherDelta();
	void clearAccumulatedDelta();
	void accumulateDelta(const std::vector<double>& delta);

	void handleParameter(const std::string & data, const RPCInfo & info);
	void waitParameter(); // waitParameter from one worker
	void checkParamChange(const Parameter& p);

// handler
public:
	void handleReply(const std::string& data, const RPCInfo& info);
	void handleOnline(const std::string& data, const RPCInfo& info);
	void handleXLength(const std::string& data, const RPCInfo& info);
	void handleDelta(const std::string& data, const RPCInfo& info);
	void handleDeltaAsync(const std::string& data, const RPCInfo& info);
	void handleDeltaFsb(const std::string& data, const RPCInfo& info);
	void handleDeltaFab(const std::string& data, const RPCInfo& info);
	void handleDeltaTail(const std::string& data, const RPCInfo& info);
	void handleDeltaDC(const std::string& data, const RPCInfo& info);
	void handleDeltaProgAsync(const std::string& data, const RPCInfo& info);
	void handleReport(const std::string& data, const RPCInfo& info);

private:
	Parameter param;
	std::vector<double> bfDelta;
	// std::vector<double> candiParam;
	std::vector< std::vector<double> > candiParam;

	IDMapper wm; // worker id mapper
	double factorDelta;
	size_t nx; // length of x // dimesion for km
	int ln; // log-every-n times
	IntervalEstimator ie; // for flexible parallel modes
	int K, D, NCnt; // for GMM

	//size_t iter; // [defined in Runner] current iteration being executate now (not complete)
	size_t nUpdate; // used for Async case
	Timer tmrTrain;
	int nIterChange;
	std::vector<double> revDelta;
	std::string staleStats;
	std::string curStats;
	int curStale;
	
	double lamda;
	size_t range;
	size_t unSendDelta, freqSendParam;
	int reportCnt, reportNum, glbBatchSize, getDeltaCnt, ttDpProcessed;
	Timer tmrDeltaV;
	std::vector<double> deltaV, deltaSS, deltaT, deltaObj;
	std::vector<int> deltaCount;
	double shrinkFactor, avgV;
	bool fastReady, factorReady, sentDReq;
	double objEsti, objImproEsti;

	size_t lastArchIter;
	Timer tmrArch;

	SyncUnit suOnline;
	SyncUnit suWorker;
	SyncUnit suAllClosed;
	SyncUnit suXLength;
	int typeDDeltaAny, typeDDeltaAll;
	SyncUnit suDeltaAny, suDeltaAll;
	SyncUnit suParam; // reply of parameter broadcast
	SyncUnit suTPause, suTContinue;

	std::ofstream foutput;
};
