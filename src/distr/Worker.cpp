#include "Worker.h"
#include "common/Option.h"
#include "network/NetworkThread.h"
#include "message/MType.h"
#include "logging/logging.h"
#include "util/Timer.h"

using namespace std;

// initialize constant / stable data
Worker::Worker() : Runner() {
	masterNID = 0;
	dataPointer = 0;
	iter = 0;
	localBatchSize = 1;

	hasNewParam = false;
	allowTrain = true;
	exitTrain = false;

	/// get from master
	typeDDeltaAny = MType::DDelta;
	typeDDeltaAll = 128 + MType::DDelta;
	//workerLst = {}; // ??
	//trainer.bindModel(&model);
	factorDelta = 1.0;
	nx = 0;
	//iter = 0;
	nUpdate = 0;
	lastArchIter = 0;
	//bufferDelta = NULL;
	bfDeltaCnt = 0;

	trainer.bindModel(&model);

	suOnline.reset();
	suParam.reset();
	suXlength.reset();

	/// for DC
	// suDeltaAny.reset();
	suDeltaAll.reset();
	// suTPause.reset();
}

void Worker::init(const Option* opt, const size_t lid)
{
	this->opt = opt;
	nWorker = opt->nw;
	localID = lid;
	trainer.setRate(opt->lrate);
	ln = opt->logIter;
	logName = "W"+to_string(localID);
	setLogThreadName(logName);

	/// for dc cache
	deltaIndx0.assign(nWorker, false);
	deltaIndx1.assign(nWorker, false);

	if(opt->mode == "sync"){
		syncInit();
	} else if(opt->mode == "async"){
		asyncInit();
	} else if(opt->mode == "fsb"){
		fsbInit();
	} else if(opt->mode == "fab"){
		fabInit();
	} else if(opt->mode == "dcsync"){
		dcSyncInit();
	} else if(opt->mode == "dcfsb"){
		dcSyncInit();
	}
}

void Worker::bindDataset(const DataHolder* pdh)
{
	VLOG(1) << "Bind dataset with " << pdh->size() << " data points";
	trainer.bindDataset(pdh);
	// separated the mini-batch among all workers
	localBatchSize = opt->batchSize / nWorker;
	if(opt->batchSize % nWorker > localID)
		++localBatchSize;
	if(localBatchSize <= 0)
		localBatchSize = 1;
}

void Worker::run()
{
	LOG(INFO) << "register handlers";
	registerHandlers();  // register message with function, function need to wrap up
	startMsgLoop(logName+"-MSG"); // make a new thread to record messages

	LOG(INFO) << "start";
	DLOG(INFO) << "send online message";
	sendOnline();

	DLOG(INFO) << "waiting worker list";
	waitWorkerList();
	DLOG(INFO) << "send x length";
	sendXLength(); // x dimension
	DLOG(INFO) << "waiting init parameter";
	waitParameter();
	DLOG(INFO) << "got init parameter";
	model.init(opt->algorighm, trainer.pd->xlength(), opt->algParam);
	applyBufferParameter();
	resumeTrain();

	DLOG(INFO) << "start training with mode: " << opt->mode << ", local batch size: " << localBatchSize;
	iter = 1;
	//try{
	//	generalProcess();
	//} catch(exception& e){
	//	LOG(FATAL) << e.what();
	//}
	if(opt->mode == "sync"){
		syncProcess();
	} else if(opt->mode == "async"){
		asyncProcess();
	} else if(opt->mode == "fsb"){
		fsbProcess();
	} else if(opt->mode == "fab"){
		fabProcess();
	} else if(opt->mode == "dcsync"){
		dcSyncProcess();
	} else if(opt->mode == "dcfsb"){
		dcFsbProcess();
	}

	DLOG(INFO) << "finish training";
	sendClosed();
	finishStat();
	showStat();
	stopMsgLoop();
}

Worker::callback_t Worker::localCBBinder(
	void (Worker::*fp)(const std::string&, const RPCInfo&))
{
	return bind(fp, this, placeholders::_1, placeholders::_2);
}

void Worker::dcSyncInit()
{
	factorDelta = 1.0 / nWorker;
	regDSPProcess(MType::DParameter, localCBBinder(&Worker::handleParameter));
	regDSPProcess(MType::DDelta, localCBBinder(&Worker::handleDelta)); /// handle delta

	// addRPHAnySU(typeDDeltaAny, suDeltaAny);
	addRPHEachSU(typeDDeltaAll, suDeltaAll);
}

void Worker::dcSyncProcess()
{	
	while(!exitTrain && iter <= opt->tcIter){
		if(allowTrain.load() == false){
			sleep();
			continue;
		}
		VLOG_EVERY_N(ln, 1) << "Iteration " << iter << ": calculate delta";
		Timer tmr;
		bfDelta = trainer.batchDelta(dataPointer, localBatchSize, true);
		updatePointer(localBatchSize);
		stat.t_dlt_calc+= tmr.elapseSd();
		VLOG_EVERY_N(ln, 2) << "  send delta";

		tmr.restart();
		broadcastDelta(bfDelta);
		bufferDelta = bfDelta;
		rph.input(typeDDeltaAll, (int)localID);

		VLOG_EVERY_N(ln, 2) << "  DC: wait for delta from all other workers";
		waitDeltaFromAll();
		stat.t_par_wait += tmr.elapseSd();

		tmr.restart();
		applyDelta();
		if(localID == 0)	/// send record to master only for worker 0
			sendParameter2M(); /// update parameter to master

		stat.t_par_calc += tmr.elapseSd();
		++iter;
	}
}

void Worker::dcFsbProcess()
{	
	while(!exitTrain){
		VLOG_EVERY_N(ln, 1) << "Iteration " << iter << ": calculate delta";
		Timer tmr;
		size_t cnt;
		
		std::vector<double> lclDelta;
		// try to use localBatchSize data-points, the actual usage is returned via cnt 
		tie(cnt, lclDelta) = trainer.batchDelta(allowTrain, dataPointer, localBatchSize, true);
		updatePointer(cnt);
		VLOG_EVERY_N(ln, 2) << "  calculate delta with " << cnt << " data points";
		stat.t_dlt_calc+= tmr.elapseSd();
		VLOG_EVERY_N(ln, 2) << "  send delta";

		tmr.restart();
		if(allowTrain == true) {
			broadcastSignalPause();
		}
		broadcastDelta(lclDelta);
		accumulateDelta(lclDelta, (int)localID);

		if(exitTrain==true){
			break;
		}
		VLOG_EVERY_N(ln, 2) << "  DC: wait for delta from all other workers";
		waitDeltaFromAll();
		if(exitTrain==true){
			break;
		}
		stat.t_par_wait += tmr.elapseSd();

		tmr.restart();
		applyDelta();
		resumeTrain();
		if(localID == 0)	/// send record to master only for worker 0
			sendParameter2M(); /// update parameter to master
		stat.t_par_calc += tmr.elapseSd();
		++iter;
	}
}

void Worker::syncInit()
{
	regDSPProcess(MType::DParameter, localCBBinder(&Worker::handleParameter));
}

void Worker::syncProcess()
{
	while(!exitTrain){
		VLOG_EVERY_N(ln, 1) << "Iteration " << iter << ": calculate delta";
		Timer tmr;
		bfDelta = trainer.batchDelta(dataPointer, localBatchSize, true);
		updatePointer(localBatchSize);
		stat.t_dlt_calc+= tmr.elapseSd();
		VLOG_EVERY_N(ln, 2) << "  send delta";
		tmr.restart();
		
		sendDelta(bfDelta);

		if(exitTrain==true){
			break;
		}
		VLOG_EVERY_N(ln, 2) << "  wait for new parameter";
		waitParameter();

		if(exitTrain==true){
			break;
		}
		stat.t_par_wait += tmr.elapseSd();
		tmr.restart();
		applyBufferParameter();
		stat.t_par_calc += tmr.elapseSd();
		++iter;
	}
}

void Worker::asyncInit()
{
	regDSPProcess(MType::DParameter, localCBBinder(&Worker::handleParameter));
}

void Worker::asyncProcess()
{
	while(!exitTrain){
		//if(allowTrain.load() == false){
		//	sleep();
		//	continue;
		//}
		VLOG_EVERY_N(ln, 1) << "Iteration " << iter << ": calculate delta";
		Timer tmr;
		bfDelta = trainer.batchDelta(dataPointer, localBatchSize, true);
		updatePointer(localBatchSize);
		stat.t_dlt_calc += tmr.elapseSd();
		VLOG_EVERY_N(ln, 2) << "  send delta";
		tmr.restart();
		sendDelta(bfDelta);
		if(exitTrain==true){
			break;
		}
		VLOG_EVERY_N(ln, 2) << "  wait for new parameter";
		waitParameter();
		if(exitTrain==true){
			break;
		}
		stat.t_par_wait += tmr.elapseSd();
		tmr.restart();
		applyBufferParameter();
		stat.t_par_calc += tmr.elapseSd();
		++iter;
	}
}

void Worker::fsbInit()
{
	regDSPProcess(MType::DParameter, localCBBinder(&Worker::handleParameter));
}

void Worker::fsbProcess()
{
	localBatchSize = trainer.pd->size();
	const size_t n = model.paramWidth();
	while(!exitTrain){
		//if(allowTrain == false){
		//	sleep();
		//	continue;
		//}
		VLOG_EVERY_N(ln, 1) << "Iteration " << iter << ": calculate delta";
		Timer tmr;
		size_t cnt = 0;
		bfDelta.assign(n, 0.0);
		while (exitTrain == false && allowTrain) {
			vector<double> tmp;
			size_t c;
			// try to use localBatchSize data-points, the actual usage is returned via cnt
			tie(c, tmp) = trainer.batchDelta(allowTrain, dataPointer, localBatchSize, true);
			accumulateDelta(tmp);
			updatePointer(c);
			cnt += c;
		}
		stat.t_dlt_calc += tmr.elapseSd();
		VLOG_EVERY_N(ln, 2) << "  calculate delta with " << cnt << " data points";
		VLOG_EVERY_N(ln, 2) << "  send delta";
		tmr.restart();
		sendDelta(bfDelta);
		if(exitTrain==true){
			break;
		}
		VLOG_EVERY_N(ln, 2) << "  wait for new parameter";
		waitParameter();
		resumeTrain();
		if(exitTrain==true){
			break;
		}
		stat.t_par_wait += tmr.elapseSd();
		tmr.restart();
		applyBufferParameter();
		resumeTrain();
		stat.t_par_calc += tmr.elapseSd();
		++iter;
	}
}

void Worker::fabInit()
{
	regDSPProcess(MType::DParameter, localCBBinder(&Worker::handleParameterFab));
}

void Worker::fabProcess()
{
	// require different handleParameter -> handleParameterFab
	const size_t n = model.paramWidth();
	const double factor = 1.0 / localBatchSize;
	while(!exitTrain){
		//if(allowTrain == false){
		//	sleep();
		//	continue;
		//}
		VLOG_EVERY_N(ln, 1) << "Iteration " << iter << ": calculate delta";// << ". msg waiting: " << driver.queSize();
		size_t left = localBatchSize;
		Timer tmr;
		bfDelta.assign(n, 0.0);
		while(!exitTrain && left != 0){
			tmr.restart();
			size_t cnt = 0;
			vector<double> tmp;
			resumeTrain();
			tie(cnt, tmp) = trainer.batchDelta(allowTrain, dataPointer, left, false);
			left -= cnt;
			updatePointer(cnt);
			//DVLOG(3) <<"tmp: "<< tmp;
			VLOG_EVERY_N(ln, 2) << "  calculate delta with " << cnt << " data points, left: " << left;
			if(cnt != 0){
				accumulateDelta(tmp);
			}
			stat.t_dlt_calc += tmr.elapseSd();
			tmr.restart();
			applyBufferParameter();
			stat.t_par_calc += tmr.elapseSd();
		}
		tmr.restart();
		for(size_t i = 0; i < n; ++i)
			bfDelta[i] *= factor;
		VLOG_EVERY_N(ln, 2) << "  send delta";
		sendDelta(bfDelta);
		// TODO: wait reply of delta for small batch size, to prevent the master from becoming a bottleneck
		// do not wait parameter, because it is loaded in each handleParameterFab
		// OR: use staleness tolerance logic in sendDelta
		if(opt->fabWait)
			waitParameter();
		stat.t_par_wait += tmr.elapseSd();
		++iter;
	}
}

void Worker::updatePointer(const size_t used)
{
	DVLOG(3) << "update pointer from " << dataPointer << " by " << used;
	dataPointer += used;
	if(dataPointer >= trainer.pd->size())
		dataPointer = 0;
	stat.n_point += used;
}

void Worker::sendOnline()
{
	net->send(masterNID, MType::COnline, localID);
}

void Worker::waitWorkerList()
{
	suOnline.wait();
}

void Worker::sendXLength()
{
	if(opt->algorighm == "km") {
		/// send k dp as candidate of global centroids
		std::vector<double> kCentroids;
		int k = stoi(opt->algParam);
		for(int i = 0; i < k; i++){
			std::vector<double> OneDp = trainer.pd->get(i).x;
			kCentroids.insert(kCentroids.end(), OneDp.begin(), OneDp.end());
			kCentroids.push_back(0); // for cluster counts
		}
		net->send(masterNID, MType::CXLength, kCentroids);
	}
	else
		net->send(masterNID, MType::CXLength, trainer.pd->xlength());
	suXlength.wait();
}

void Worker::sendClosed()
{
	net->send(masterNID, MType::CClosed, localID);
}

void Worker::registerHandlers()
{
	regDSPProcess(MType::CReply, localCBBinder(&Worker::handleReply));
	regDSPProcess(MType::CWorkers, localCBBinder(&Worker::handleWorkerList));
	regDSPProcess(MType::CTrainPause, localCBBinder(&Worker::handlePause));
	regDSPProcess(MType::CTrainContinue, localCBBinder(&Worker::handleContinue));
	regDSPImmediate(MType::CTerminate, localCBBinder(&Worker::handleTerminate));

	//regDSPProcess(MType::DParameter, localCBBinder(&Worker::handleParameter));
	//regDSPProcess(MType::DDelta, localCBBinder(&Worker::handleDelta));

	//addRPHAnySU(MType::CWorkers, suOnline);
	//addRPHAnySU(MType::DParameter, suParam);
	addRPHAnySU(MType::CXLength, suXlength);
}

void Worker::accumulateDelta(const std::vector<double>& delta)
{
	for (size_t i = 0; i < delta.size(); ++i)
		bfDelta[i] += delta[i];
}

void Worker::sendDelta(std::vector<double>& delta)
{
	// TODO: add staleness tolerance logic here
	DVLOG(3) << "send delta: " << delta;
	//DVLOG_EVERY_N(ln, 1) << "n-send: " << iter << " un-cmt msg: " << net->pending_pkgs() << " cmt msg: " << net->stat_send_pkg;
	net->send(masterNID, MType::DDelta, delta);
	++stat.n_dlt_send;
}

void Worker::broadcastDelta(std::vector<double>& delta)
{
	// TODO: add staleness tolerance logic here
	DVLOG(3) << "broadcast delta: " << delta;
	//DVLOG_EVERY_N(ln, 1) << "n-send: " << iter << " un-cmt msg: " << net->pending_pkgs() << " cmt msg: " << net->stat_send_pkg;
	///// for(int lid; lid < )
	net->broadcast(MType::DDelta, delta);
	++stat.n_dlt_send;
}

void Worker::bufferParameter(Parameter & p)
{
	lock_guard<mutex> lk(mParam);
	bfParam = move(p);
	hasNewParam = true;
}

void Worker::applyBufferParameter()
{
	//DLOG(INFO)<<"has new parameter: "<<hasNewParam;
	if(!hasNewParam)
		return;
	//DLOG(INFO)<<"before lock";
	//lock(mParam, mModel);
	lock_guard<mutex> lk(mParam);
	//DLOG(INFO)<<"after lock";
	DVLOG(3) << "apply parameter: " << bfParam.weights;
	model.setParameter(bfParam);
	//mModel.unlock();
	hasNewParam = false;
	//mParam.unlock();
}

void Worker::waitParameter()
{
	suParam.wait();
	suParam.reset();
}

void Worker::fetchParmeter()
{
	net->send(masterNID, MType::DRParameter, localID);
	suParam.wait();
	++stat.n_dlt_recv;
}

void Worker::pauseTrain()
{
	// if(bfDelta != NULL)
		allowTrain = false;
}

void Worker::resumeTrain()
{
	allowTrain = true;
}

void Worker::handleReply(const std::string& data, const RPCInfo& info) {
	Timer tmr;
	int type = deserialize<int>(data);
	stat.t_data_deserial += tmr.elapseSd();
	pair<bool, int> s = wm.nidTrans(info.source);
	DVLOG(4) << "get reply from " << (s.first ? "W" : "M") << s.second << " type " << type;
	/*static int ndr = 0;
	if(type == MType::DDelta){
		++ndr;
		VLOG_EVERY_N(ln / 10, 1) << "get delta reply: " << ndr;
	}*/
	rph.input(type, s.second);
}

void Worker::handleWorkerList(const std::string & data, const RPCInfo & info)
{
	DLOG(INFO) << "receive worker list";
	Timer tmr;
	auto res = deserialize<vector<pair<int, int>>>(data);
	stat.t_data_deserial += tmr.elapseSd();
	for(auto& p : res){
		DLOG(INFO)<<"register nid "<<p.first<<" with lid "<<p.second;
		wm.registerID(p.first, p.second);
	}
	//rph.input(MType::CWorkers, info.source);
	suOnline.notify();
	sendReply(info);
}

void Worker::handleParameter(const std::string & data, const RPCInfo & info)
{
	Timer tmr;
	auto weights = deserialize<vector<double>>(data);
	stat.t_data_deserial += tmr.elapseSd();
	Parameter p;
	p.set(move(weights));
	bufferParameter(p);
	suParam.notify();
	//sendReply(info);
	++stat.n_par_recv;
}

void Worker::handleParameterFab(const std::string & data, const RPCInfo & info)
{
	Timer tmr;
	auto weights = deserialize<vector<double>>(data);
	stat.t_data_deserial += tmr.elapseSd();
	Parameter p;
	p.set(move(weights));
	bufferParameter(p);
	suParam.notify();
	//sendReply(info);
	// break the trainning and apply the received parameter (in main thread)
	pauseTrain();
	//applyBufferParameter();
	++stat.n_par_recv;
}

void Worker::handlePause(const std::string & data, const RPCInfo & info)
{
	pauseTrain();
	sendReply(info);
}

void Worker::handleContinue(const std::string & data, const RPCInfo & info)
{
	resumeTrain();
	sendReply(info);
}

void Worker::handleTerminate(const std::string & data, const RPCInfo & info)
{
	exitTrain = true;
	pauseTrain(); // in case if the system is calculating delta
	suParam.notify(); // in case if the system just calculated a delta (is waiting for new parameter)
	suDeltaAll.notify(); // in case the worker is waiting other parameters
	sendReply(info);
}

//  how worker handle delta
//
void Worker::handleDelta(const std::string & data, const RPCInfo & info)
{
	auto delta = deserialize<vector<double>>(data);
	int s = wm.nid2lid(info.source);
	// rph.input(typeDDeltaAll, s);
	accumulateDelta(delta, s);
///	applyDelta(delta, s);
///	rph.input(typeDDeltaAll, s);
///	rph.input(typeDDeltaAny, s);
	//sendReply(info);
	++stat.n_dlt_recv;
}

void Worker::waitDeltaFromAll()
{
	suDeltaAll.wait();
	suDeltaAll.reset();
}

void Worker::accumulateDelta(std::vector<double>& delta, const int source)
{
	lock_guard<mutex> lk(mDelta);
	if (deltaIndx0[source]) { // if a delta from source is already there
		copyDelta(bufferDeltaExt, delta);
		deltaIndx1[source] = true;
		bfDeltaCnt++;
		// DVLOG_IF(bfDeltaCnt > nWorker, 2) << " Dam MORE number of delta applied &&&&&&&";
	}
	else {
		DVLOG_IF(deltaIndx1[source], 2) << " Dam WWWTTTFFFF number of delta applied &&&&&&&";
		copyDelta(bufferDelta, delta);
		deltaIndx0[source] = true;
		rph.input(typeDDeltaAll, source); // trigger the syncUnit counter
	}
}

void Worker::copyDelta(std::vector<double>& buffer, std::vector<double>& delta){

	if(buffer.empty()) {
		buffer = move(delta);
	}
	else {
		for(int i = 0; i < delta.size(); i++)
			buffer[i] += delta[i];
	}
}

void Worker::applyDelta()
{
	DVLOG(3) << "apply buffered delta : " << bufferDelta
		<< "\nonto: " << model.getParameter().weights;
	model.accumulateParameter(bufferDelta, factorDelta);

	/// resetDcBuffer
	DVLOG_IF(bfDeltaCnt > 0, 3) << "reset buffered delta index: " << deltaIndx1
		<< "\nto: " << deltaIndx0;
	bufferDelta = move(bufferDeltaExt);
	bufferDeltaExt.clear();
	for(int i = 0; i < deltaIndx1.size(); i++){
		if(deltaIndx1[i]){
			rph.input(typeDDeltaAll, i); // add accumulated syncUnit counter
		}
	}
	deltaIndx0 = move(deltaIndx1);
	deltaIndx1.assign(nWorker, false);
	bfDeltaCnt = 0;
}

void Worker::sendParameter2M()
{
	DVLOG(3) << "send parameter to master with: " << model.getParameter().weights;
	net->send(masterNID, MType::DParameter, model.getParameter().weights);
	++stat.n_par_send;
}

void Worker::broadcastSignalPause()
{
	net->broadcast(MType::CTrainPause, "");
	// suTPause.wait();
	// suTPause.reset();
}
