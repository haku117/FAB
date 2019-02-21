#include "Master.h"
#include "common/Option.h"
#include "network/NetworkThread.h"
#include "message/MType.h"
#include "logging/logging.h"
using namespace std;

Master::Master() : Runner() {
	typeDDeltaAny = MType::DDelta;
	typeDDeltaAll = 128 + MType::DDelta;
	
	factorDelta = 1.0;
	nx = 0;
	// candiParam;
	iter = 0;
	nUpdate = 0;
	nIterChange = 0;
	lastArchIter = 0;
	tmrArch.restart();

	suOnline.reset();
	suWorker.reset();
	suXLength.reset();
	suParam.reset();
	suDeltaAny.reset();
	suDeltaAll.reset();
	suTPause.reset();
	suTContinue.reset();
	suAllClosed.reset();
}

void Master::init(const Option* opt, const size_t lid)
{
	this->opt = opt;
	nWorker = opt->nw;
	trainer.setRate(opt->lrate);
	localID = lid;
	ln = opt->logIter;
	logName = "M";
	setLogThreadName(logName);
	if(opt->mode == "sync"){
		syncInit();
	} else if(opt->mode == "async"){
		asyncInit();
	} else if(opt->mode == "fsb"){
		fsbInit();
		// TODO: add specific option for interval estimator
		// ie.init(nWorker, { "fixed", to_string(opt->arvTime/opt->batchSize) });
	} else if(opt->mode == "fab"){
		fabInit();
	} else if(opt->mode == "dcsync"){
		dcInit();
	} else if(opt->mode == "dcfsb"){
		dcInit();
	}
}

void Master::run()
{
	registerHandlers();
	startMsgLoop(logName+"-MSG");
	
	LOG(INFO) << "Wait online messages";
	suOnline.wait();
	LOG(INFO) << "Send worker list";
	broadcastWorkerList();
	LOG(INFO)<<"Waiting x-length to initialize parameters";
	initializeParameter();
	trainer.bindModel(&model);/// move
	
	clearAccumulatedDelta();
	LOG(INFO) << "Got x-length = " << nx;
	if(!opt->fnOutput.empty()){
		foutput.open(opt->fnOutput);
		LOG_IF(foutput.fail(), FATAL) << "Cannot write to file: " << opt->fnOutput;
	}
	iter = 0;
	tmrTrain.restart();
	archiveProgress(true);
	LOG(INFO) << "Broadcasting initial parameter";
	broadcastParameter();

	LOG(INFO)<<"Start traning with mode: "<<opt->mode;
	//tmrTrain.restart();
	iter = 1;
	if(opt->mode == "sync"){
		syncProcess();
	} else if(opt->mode == "async"){
		asyncProcess();
	} else if(opt->mode == "fsb"){
		fsbProcess();
	} else if(opt->mode == "fab"){
		fabProcess();
	} else if(opt->mode == "dcsync"){
		dcProcess();
	} else if(opt->mode == "dcfsb"){
		dcProcess();
	}

	double t = tmrTrain.elapseSd();
	LOG(INFO) << "Finish training. Time cost: " << t << ". Iterations: " << iter
		<< ". Average iteration time: " << t / iter;

	broadcastSignalTerminate();
	regDSPProcess(MType::DDelta, localCBBinder(&Master::handleDeltaTail));
	foutput.close();
	DLOG(INFO) << "un-send: " << net->pending_pkgs() << ", un-recv: " << net->unpicked_pkgs();
	finishStat();
	showStat();
	suAllClosed.wait();
	stopMsgLoop();
}

Master::callback_t Master::localCBBinder(
	void (Master::*fp)(const std::string&, const RPCInfo&))
{
	return bind(fp, this, placeholders::_1, placeholders::_2);
}

void Master::dcInit()
{
	regDSPProcess(MType::DDelta, localCBBinder(&Master::handleDeltaDC));
}

void Master::dcProcess()
{
	double tl = tmrTrain.elapseSd();
	while(!terminateCheck()){
		if(VLOG_IS_ON(2) && iter % 100 == 0){
			double t = tmrTrain.elapseSd();
			VLOG(2) << "  Average iteration time of recent 100 iterations: " << (t - tl) / 100;
			tl = t;
		}
		VLOG_EVERY_N(ln, 1)<<"Start iteration: "<<iter;
///		waitDeltaFromAll();
		waitParameter(); // wait one parameter update

		VLOG_EVERY_N(ln, 2) << "  DC: receive new parameters";
///		broadcastParameter();
		archiveProgress();
		//waitParameterConfirmed();
		++iter;
	}
}

void Master::syncInit()
{
	factorDelta = 1.0; // nWorker;
	regDSPProcess(MType::DDelta, localCBBinder(&Master::handleDelta));
}

void Master::syncProcess()
{
	double tl = tmrTrain.elapseSd();
	while(!terminateCheck()){
		if(VLOG_IS_ON(2) && iter % 100 == 0){
			double t = tmrTrain.elapseSd();
			VLOG(2) << "  Average iteration time of recent 100 iterations: " << (t - tl) / 100;
			tl = t;
		}
		VLOG_EVERY_N(ln, 1)<<"Start iteration: "<<iter << "\t" << nIterChange;
		waitDeltaFromAll();

		VLOG_EVERY_N(ln, 2) << "  Broadcast new parameters";
		broadcastParameter();
		archiveProgress();
		//waitParameterConfirmed();
		++iter;
	}
}

void Master::asyncInit()
{
	factorDelta = 1.0;
	regDSPProcess(MType::DDelta, localCBBinder(&Master::handleDeltaAsync));
}

void Master::asyncProcess()
{
	bool newIter = true;
	double tl = tmrTrain.elapseSd();
	while(!terminateCheck()){
		if(newIter){
			VLOG_EVERY_N(ln, 1) << "Start iteration: " << iter;
			newIter = false;
			if(VLOG_IS_ON(2) && iter % 100 == 0){
				double t = tmrTrain.elapseSd();
				VLOG(2) << "  Average iteration time of recent 100 iterations: " << (t - tl) / 100;
				tl = t;
			}
		}
		VLOG_EVERY_N(ln, 2) << "In iteration: " << iter << " update: " << nUpdate;
		waitDeltaFromAny();
		suDeltaAny.reset();
		size_t p = nUpdate / nWorker + 1;
		if(iter != p){
			archiveProgress();
			iter = p;
			newIter = true;
		}
	}
}

void Master::fsbInit()
{
	factorDelta = 1.0; // nWorker;
	// regDSPProcess(MType::DDelta, localCBBinder(&Master::handleDeltaFsb));
	regDSPProcess(MType::DDelta, localCBBinder(&Master::handleDelta));
}

void Master::fsbProcess()
{
	double tl = tmrTrain.elapseSd();
	while(!terminateCheck()){
		if(VLOG_IS_ON(2) && iter % 100 == 0){
			double t = tmrTrain.elapseSd();
			VLOG(2) << "  Average iteration time of recent 100 iterations: " << (t - tl) / 100;
			tl = t;
		}
		VLOG_EVERY_N(ln, 1)<<"Start iteration: "<<iter;
		waitDeltaFromAny();
		// double interval = ie.interval();
		// sleep(interval);
		VLOG_EVERY_N(ln, 2) << "  Broadcast pause signal";
		broadcastSignalPause();
		VLOG_EVERY_N(ln, 2) << "  Waiting for all deltas";
		waitDeltaFromAll();
		VLOG_EVERY_N(ln, 2) << "  Broadcast new parameters";
		broadcastParameter();
		//waitParameterConfirmed();
		// ie.update(bfDelta, interval, tmrTrain.elapseSd());
		//VLOG_EVERY_N(ln, 2) << "  Broadcast continue signal";
		//broadcastSignalContinue();
		archiveProgress();
		++iter;
	}
}

void Master::fabInit()
{
	factorDelta = 1.0;
	regDSPProcess(MType::DDelta, localCBBinder(&Master::handleDeltaFab));
}

void Master::fabProcess()
{
	bool newIter = true;
	double tl = tmrTrain.elapseSd();
	while(!terminateCheck()){
		if(newIter){
			VLOG_EVERY_N(ln, 1) << "Start iteration: " << iter;// << ", msg waiting: " << driver.queSize() << ", update: " << nUpdate;
			DVLOG_EVERY_N(ln / 10, 1) << "un-send: " << net->pending_pkgs() << ", un-recv: " << net->unpicked_pkgs();
			newIter = false;
			if(VLOG_IS_ON(2) && iter % 100 == 0){
				double t = tmrTrain.elapseSd();
				VLOG(2) << "  Time of recent 100 iterations: " << (t - tl);
				tl = t;
			}
		}
		VLOG_EVERY_N(ln, 2) << "In iteration: " << iter << " update: " << nUpdate;
		waitDeltaFromAny();
		suDeltaAny.reset();
		broadcastParameter();
		size_t p = nUpdate / nWorker + 1;
		if(iter != p){
			archiveProgress();
			iter = p;
			newIter = true;
		}
	}
}

void Master::registerHandlers()
{
	regDSPProcess(MType::CReply, localCBBinder(&Master::handleReply));
	regDSPProcess(MType::COnline, localCBBinder(&Master::handleOnline));
	regDSPProcess(MType::CXLength, localCBBinder(&Master::handleXLength));
	// regDSPProcess(MType::DDelta, localCBBinder(&Master::handleDelta)); // for sync and fsb
	// regDSPProcess(MType::DDelta, localCBBinder(&Master::handleDeltaAsync)); // for async

	regDSPProcess(MType::DParameter, localCBBinder(&Master::handleParameter));

	addRPHEachSU(MType::COnline, suOnline);
	addRPHEachSU(MType::CWorkers, suWorker);
	addRPHEachSU(MType::CXLength, suXLength);
	addRPHEachSU(MType::DParameter, suParam);
	addRPHEachSU(MType::CTrainPause, suTPause);
	addRPHEachSU(MType::CTrainContinue, suTContinue);
	addRPHEachSU(MType::CTerminate, suAllClosed);
	addRPHAnySU(typeDDeltaAny, suDeltaAny);
	addRPHEachSU(typeDDeltaAll, suDeltaAll);
}

void Master::bindDataset(const DataHolder* pdh)
{
	trainer.bindDataset(pdh);
}

void Master::applyDelta(std::vector<double>& delta, const int source)
{
	DVLOG(3) << "apply delta from " << source << " : " << delta
		<< "\nonto: " << model.getParameter().weights;
	model.accumulateParameter(delta, factorDelta);
}

bool Master::terminateCheck()
{
	return (iter >= opt->tcIter)
		|| (tmrTrain.elapseSd() > opt->tcTime
		|| (nIterChange > 10));
}

void Master::initializeParameter()
{
	suXLength.wait();
	suXLength.reset();
	if(opt->algorighm == "km") {
		model.init(opt->algorighm, nx, opt->algParam, candiParam);
	}
	else
		model.init(opt->algorighm, nx, opt->algParam, 0.01);
}

void Master::sendParameter(const int target)
{
	DVLOG(3) << "send parameter to " << target << " with: " << model.getParameter().weights;
	net->send(wm.lid2nid(target), MType::DParameter, model.getParameter().weights);
	++stat.n_par_send;
}

void Master::broadcastParameter()
{
	DVLOG(3) << "broad parameter: " << model.getParameter().weights;
	net->broadcast(MType::DParameter, model.getParameter().weights);
	stat.n_par_send += nWorker;
}

void Master::waitParameterConfirmed()
{
	suParam.wait();
	suParam.reset();
}

bool Master::needArchive()
{
	if(!foutput.is_open())
		return false;
	if(iter - lastArchIter >= opt->arvIter
		|| tmrArch.elapseSd() >= opt->arvTime)
	{
		lastArchIter = iter;
		tmrArch.restart();
		return true;
	}
	return false;
}

void Master::archiveProgress(const bool force)
{
	if(!force && !needArchive())
		return;
	foutput << iter << "," << tmrTrain.elapseSd();
	for(auto& v : model.getParameter().weights){
		foutput << "," << v;
	}
	foutput <<"\n";
}

void Master::broadcastWorkerList()
{
	vector<pair<int, int>> temp = wm.list();
	net->broadcast(MType::CWorkers, temp);
	suWorker.wait();
}

void Master::broadcastSignalPause()
{
	net->broadcast(MType::CTrainPause, "");
	suTPause.wait();
	suTPause.reset();
}

void Master::broadcastSignalContinue()
{
	net->broadcast(MType::CTrainContinue, "");
	suTContinue.wait();
	suTContinue.reset();
}

void Master::broadcastSignalTerminate()
{
	net->broadcast(MType::CTerminate, "");
}

void Master::waitDeltaFromAny(){
	suDeltaAny.wait();
}

void Master::waitDeltaFromAll(){
	suDeltaAll.wait();
	suDeltaAll.reset();
}

void Master::gatherDelta()
{
	suDeltaAll.reset();
	net->broadcast(MType::DRDelta, "");
	suDeltaAll.wait();
}

void Master::clearAccumulatedDelta()
{
	bfDelta.assign(model.paramWidth(), 0.0);
}

void Master::accumulateDelta(const std::vector<double>& delta)
{
	for (size_t i = 0; i < delta.size(); ++i)
		bfDelta[i] += delta[i];
}

void Master::handleReply(const std::string & data, const RPCInfo & info)
{
	Timer tmr;
	int type = deserialize<int>(data);
	stat.t_data_deserial += tmr.elapseSd();
	int source = wm.nid2lid(info.source);
	rph.input(type, source);
}

void Master::handleOnline(const std::string & data, const RPCInfo & info)
{
	Timer tmr;
	auto lid = deserialize<int>(data);
	stat.t_data_deserial += tmr.elapseSd();
	wm.registerID(info.source, lid);
	rph.input(MType::COnline, lid);
	sendReply(info);
}

void Master::handleXLength(const std::string& data, const RPCInfo& info){
	Timer tmr;
	int source = wm.nid2lid(info.source);

	if(opt->algorighm == "km") {
		std::vector<double> centroids = deserialize<std::vector<double>>(data);
		stat.t_data_deserial += tmr.elapseSd();
		int k = stoi(opt->algParam);
		if(nx == 0){
			nx = centroids.size()/k;
		} else if(nx != centroids.size()/k){
			LOG(FATAL)<<"dataset on "<<source<<" does not match with others";
		}
		if (candiParam.empty()){
			candiParam = move(centroids);
		}
	} 
	else {
		size_t d = deserialize<size_t>(data);
		stat.t_data_deserial += tmr.elapseSd();
		if(nx == 0){
			nx = d;
		} else if(nx != d){
			LOG(FATAL)<<"dataset on "<<source<<" does not match with others";
		}
	}
	rph.input(MType::CXLength, source);
	sendReply(info);
}

void Master::handleDelta(const std::string & data, const RPCInfo & info)
{
	Timer tmr;
	auto delta = deserialize<vector<double>>(data);
	stat.t_data_deserial += tmr.elapseSd();
	int s = wm.nid2lid(info.source);
	applyDelta(delta, s);
	rph.input(typeDDeltaAll, s);
	rph.input(typeDDeltaAny, s);
	//sendReply(info);
	++stat.n_dlt_recv;
}

void Master::handleDeltaAsync(const std::string & data, const RPCInfo & info)
{
	Timer tmr;
	auto delta = deserialize<vector<double>>(data);
	stat.t_data_deserial += tmr.elapseSd();
	int s = wm.nid2lid(info.source);
	applyDelta(delta, s);
	++nUpdate;
	rph.input(typeDDeltaAll, s);
	rph.input(typeDDeltaAny, s);
	//sendReply(info);
	++stat.n_dlt_recv;
	// directly send new parameter
	sendParameter(s);
}

void Master::handleDeltaFsb(const std::string & data, const RPCInfo & info)
{
	Timer tmr;
	auto delta = deserialize<vector<double>>(data);
	stat.t_data_deserial += tmr.elapseSd();
	int s = wm.nid2lid(info.source);
	accumulateDelta(delta);
	applyDelta(delta, s);
	rph.input(typeDDeltaAll, s);
	//rph.input(typeDDeltaAny, s);
	//sendReply(info);
	++stat.n_dlt_recv;
}

void Master::handleDeltaFab(const std::string & data, const RPCInfo & info)
{
	Timer tmr;
	auto delta = deserialize<vector<double>>(data);
	stat.t_data_deserial += tmr.elapseSd();
	int s = wm.nid2lid(info.source);
	applyDelta(delta, s);
	++nUpdate;
	//static vector<int> cnt(nWorker, 0);
	//++cnt[s];
	//VLOG_EVERY_N(ln/10, 1) << "Update: " << nUpdate << " rsp: " << cnt << " r-pkg: " << net->stat_recv_pkg;
	rph.input(typeDDeltaAll, s);
	rph.input(typeDDeltaAny, s);
	if(opt->fabWait)
		sendReply(info);
	++stat.n_dlt_recv;
	// broadcast new parameter in main thread
}

void Master::handleDeltaTail(const std::string & data, const RPCInfo & info)
{	
	if(opt->algorighm != "km") {
		Timer tmr;
		auto delta = deserialize<vector<double>>(data);
		stat.t_data_deserial += tmr.elapseSd();
		int s = wm.nid2lid(info.source);
		// DVLOG(3) << " call apply delta from handleDeltaTail"; 
		applyDelta(delta, s);
	}
	++stat.n_dlt_recv;
}

void Master::handleDeltaDC(const std::string & data, const RPCInfo & info)
{
	++stat.n_dlt_recv;
}

/// update parameter from worker (DC)
void Master::handleParameter(const std::string & data, const RPCInfo & info)
{
	auto weights = deserialize<vector<double>>(data);
	Parameter p;
	p.set(move(weights));
	DVLOG(3) << "apply parameter: " << p.weights;
	checkParamChange(p);
	model.setParameter(p);
	suParam.notify();
	//sendReply(info);
	++stat.n_par_recv;
}

void Master::checkParamChange(const Parameter& p){
	if(param.isSameParm(p)) {
		nIterChange++;
	}
	else {
		nIterChange = 0;
	}
	
}

void Master::waitParameter()
{
	suParam.wait();
	suParam.reset();
}
