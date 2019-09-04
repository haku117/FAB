#include "Master.h"
#include "common/Option.h"
#include "network/NetworkThread.h"
#include "message/MType.h"
#include "logging/logging.h"
#include "util/Util.h"
using namespace std;

Master::Master() : Runner() {
	typeDDeltaAny = MType::DDelta;
	typeDDeltaAll = 128 + MType::DDelta;
	
	factorDelta = 1.0;
	shrinkFactor = 1.0;
	nx = 0;
	// candiParam;
	iter = 0;
	nUpdate = 0;
	nIterChange = 0;
	lastArchIter = 0;
	revDelta = std::vector<double>();
	tmrArch.restart();
	staleStats = "";
	objEsti = 0;
	objImproEsti = 0.0;

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
	localID = lid;
	ln = opt->logIter;
	logName = "M";
	curStats = "";
	unSendDelta = 0;
	freqSendParam = 1;
	reportCnt = 0;
	reportNum = 0;
	glbBatchSize = opt->batchSize;
	getDeltaCnt = 0;
	ttDpProcessed = 0;
	NCnt = 0;
	D = -1;
	deltaV.assign(nWorker+3, 0);
	deltaCount.assign(nWorker, 0);
	deltaT.assign(nWorker, 0.0);
	deltaObj.assign(nWorker, 0.0);

	fastReady = false;
	factorReady = false;
	sentDReq = false;
	avgV = 9999;

	if (opt->algorighm == "km" || opt->algorighm == "nmf"
		 || opt->algorighm == "lda" || opt->algorighm == "gmm") {
		trainer = new EM;
		std::vector<int> tokens = parseParam(opt->algParam);
		K = tokens[0];
		// if (tokens.size() > 3){
		// 	// lamda = double(tokens[1])/100;
		// 	// range = tokens[2];
		// 	freqSendParam = tokens[3];
		// }

		if (opt->mode.find("sm") !=std::string::npos) glbBatchSize *= 0.9;
		if (opt->mode.find("lg") !=std::string::npos) glbBatchSize *= 1.1;

	} else {
		trainer = new GD;
		trainer->setRate(opt->lrate);
		// if (opt->algorighm == "mlp"){
		// 	std::vector<int> tokens = parseParam(opt->algParam);
		// 	int tn = tokens.size();
		// 	freqSendParam = tokens[tn-1];
		// }
	}
	VLOG(1) << "FPM: " << opt->reptr;

	setLogThreadName(logName);
	if(opt->mode.find("sync") !=std::string::npos){
		syncInit();
	} else if(opt->mode.find("asyc") !=std::string::npos){
		asyncInit();
	} else if(opt->mode.find("pasp") !=std::string::npos){
		progAsyncInit();
	}else if(opt->mode == "fsb"){
		fsbInit();
		// TODO: add specific option for interval estimator
		// ie.init(nWorker, { "fixed", to_string(opt->arvTime/opt->batchSize) });
	} else if(opt->mode == "fab"){
		fabInit();
	} else if(opt->mode == "dcsync"){
		dcInit();
	} else if(opt->mode.find("dc") !=std::string::npos){
		dcInit();
	}else if(opt->mode.find("pipe") !=std::string::npos){
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
	trainer->bindModel(&model);/// move
	
	clearAccumulatedDelta();
	LOG(INFO) << "Got x-length = " << nx;
	if(!opt->fnOutput.empty()){
		foutput.open(opt->fnOutput);
		LOG_IF(foutput.fail(), FATAL) << "Cannot write to file: " << opt->fnOutput;
	}
	iter = 0;
	tmrTrain.restart();
	if(opt->mode.find("sync") !=std::string::npos)
		archiveProgress(true);
	else
		archiveProgressAsync("0", true);
	LOG(INFO) << "Broadcasting initial parameter";
	bool version = false;
	if (opt->mode.find("asyc") !=std::string::npos)
		version = true;
	broadcastParameter(version);

	LOG(INFO)<<"Start traning with mode: "<<opt->mode;
	tmrTrain.restart();
	iter = 1;
	if(opt->mode.find("sync") !=std::string::npos){
		syncProcess();
	} else if(opt->mode.find("asyc") !=std::string::npos){
		asyncProcess();
	} else if(opt->mode.find("pasp") !=std::string::npos){
		progAsyncProcess();
	}else if(opt->mode == "fsb"){
		fsbProcess();
	} else if(opt->mode == "fab"){
		fabProcess();
	} else if(opt->mode.find("dc") !=std::string::npos){
		dcProcess();
	} else if(opt->mode.find("pipe") !=std::string::npos){
		dcProcess();
	} 
	// else if(opt->mode == "dcfsb"){
	// 	dcProcess();
	// }

	double t = tmrTrain.elapseSd();
	LOG(INFO) << "Finish training. Time cost: " << t << ". Iterations: " << iter
		<< ". Average iteration time: " << t / iter;
	LOG(INFO) << "Staleness:\t" << staleStats;
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
	regDSPProcess(MType::DDeltaRPL, localCBBinder(&Master::handleDeltaDC));
}

void Master::dcProcess()
{
	double tl = tmrTrain.elapseSd();
	size_t arcIter = 0;
	while(!terminateCheck()){
		if(VLOG_IS_ON(2) && iter % 100 == 0){
			double t = tmrTrain.elapseSd();
			VLOG(2) << "  Average iteration time of recent 100 iterations: " << (t - tl) / 100;
			tl = t;
		}
		VLOG_EVERY_N(ln, 1)<<"Start iteration: "<<iter;
		// VLOG(2) << "Master wait for new param";
		waitParameter(); // wait one parameter update
		// VLOG(2) << "Master received new param";

		VLOG_EVERY_N(ln, 3) << "  DC: receive new parameters";
		if(archiveProgress())
			arcIter = iter;
		//waitParameterConfirmed();
		++iter;
	}
	if (iter > arcIter + 2)
		archiveProgress(true);
}

void Master::syncInit()
{
	factorDelta = 1.0; // nWorker;
	regDSPProcess(MType::DDelta, localCBBinder(&Master::handleDelta));
}

void Master::syncProcess()
{
	double tl = tmrTrain.elapseSd();
	if (opt->algorighm == "lda") {
		int prev = model.paramWidth();
		model.resetparam();
		VLOG(1) << "reinit param from: " << prev << " to: " << model.paramWidth();
	}
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
	if (opt->algorighm == "lda")
		model.resetparam();
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
			// archiveProgress();
			iter = p;
			newIter = true;
		}
	}
}

void Master::progAsyncInit()
{
	factorDelta = 1.0;
	regDSPProcess(MType::DDelta, localCBBinder(&Master::handleDeltaProgAsync));
	regDSPProcess(MType::DReport, localCBBinder(&Master::handleReport));
}

void Master::progAsyncProcess()
{
	bool newIter = true;
	double tl = tmrTrain.elapseSd();
	tmrDeltaV.restart();
	if (opt->algorighm == "lda")
		model.resetparam();
	while(!terminateCheck()){
		// VLOG_EVERY_N(ln, 2) << "In iteration: " << iter << " update: " << nUpdate;
		VLOG_EVERY_N(20, 1) << "In iteration: " << iter << " update: " << nUpdate 
			<< " dp: " << ttDpProcessed;

		if(opt->mode.find("pasp5") !=std::string::npos){
			waitDeltaFromAll();
			ttDpProcessed += getDeltaCnt;
			broadcastParameter();
			nUpdate++;
			VLOG_IF(nUpdate < 3, 1) << "pasp5 broadcastParameter: " << iter << " update: " << nUpdate 
				<< " dp: " << ttDpProcessed;
			archiveProgressAsync(std::to_string(objImproEsti/getDeltaCnt), false);
			getDeltaCnt = 0;
		} 
		else {
			waitDeltaFromAny();
			suDeltaAny.reset();
		}

		if(unSendDelta == 0){
			++iter;
		}
	}
}

/*** void Master::progAsyncProcess()
{
	bool newIter = true;
	double tl = tmrTrain.elapseSd();
	if (opt->algorighm == "lda")
		model.resetparam();
	while(!terminateCheck()){
		// if(newIter){
		// 	VLOG_EVERY_N(ln, 1) << "Start iteration: " << iter;
		// 	newIter = false;
		// 	if(VLOG_IS_ON(2) && iter % 100 == 0){
		// 		double t = tmrTrain.elapseSd();
		// 		VLOG(2) << "  Average iteration time of recent 100 iterations: " << (t - tl) / 100;
		// 		tl = t;
		// 	}
		// }
		VLOG_EVERY_N(20, 1) << "In iteration: " << iter << " update: " << nUpdate;
		waitDeltaFromAny();
		suDeltaAny.reset();
		// size_t p = nUpdate / freqSendParam;
		if(unSendDelta == 0){
			// archiveProgress();
			++iter;
			// newIter = true;
		}
	}
} 
void Master::handleDeltaProgAsync(const std::string & data, const RPCInfo & info)
{
	Timer tmr;
	int s = wm.nid2lid(info.source);
	auto delta = deserialize<vector<double>>(data);
	size_t dpCnt = delta.back();
	delta.pop_back();

	stat.t_data_deserial += tmr.elapseSd();
	applyDelta(delta, s);

	rph.input(typeDDeltaAll, s);
	rph.input(typeDDeltaAny, s);
	//sendReply(info);
	++stat.n_dlt_recv;
	// directly send new parameter
	// sendParameter(s);
	++unSendDelta;
	if (unSendDelta >= freqSendParam){
		broadcastParameter();
		++nUpdate;
		unSendDelta = 0;
		archiveProgressAsync("", true);
	}
}***/

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
	trainer->bindDataset(pdh);
}

void Master::applyDelta(std::vector<double>& delta, const int source)
{
	DVLOG(3) << "apply delta from " << source << " : " << delta.size() << "; " << delta
		<< "\nonto: " << model.getParameter().size() << "; " << model.getParameter().weights;
	// if (opt->algorighm == "lda"){
	// 	model.accumulateParameterLDA(delta, stoi(opt->algParam) );
	// }
	model.accumulateParameter(delta, factorDelta);
}

bool Master::terminateCheck()
{
	//return (iter >= opt->tcIter)
	return (nUpdate >= opt->tcIter or iter >= opt->tcIter ) // for km pasp
		|| (tmrTrain.elapseSd() > opt->tcTime
		|| (nIterChange > 10));
}

void Master::initializeParameter()
{
	suXLength.wait();
	suXLength.reset();
	if(opt->algorighm == "km") {
		vector<double> param = candiParam[0];
		for (int i = 1; i < candiParam.size(); i++) {
			param.insert(param.end(), candiParam[i].begin(), candiParam[i].end());
		}
		model.init(opt->algorighm, nx, opt->algParam, param);
	} 
	else if(opt->algorighm == "gmm") {

		VLOG(3) << "GMM init param: nx:" << nx << " K:" << K 
				<< ", candiSize:" << candiParam[0].size();

		// candiParam[0].pop_back();
		vector<double> param = candiParam[0];
		for (int i = 1; i < candiParam.size(); i++) {
			param.insert(param.end(), candiParam[i].begin(), candiParam[i].end()); /// for mean
			// param.pop_back();
		}
		for (int k = 0; k < K; k++){
			for (int d1 = 0; d1 < nx; d1++){
				for (int d2 = 0; d2 < nx; d2++){
					param.push_back(d1 == d2); /// for covariance
				}
			}
		}
		for (int k = 0; k < K; k++){
			param.push_back(1.0/K); /// for weights
		}
		
		model.init(opt->algorighm, nx, opt->algParam, param);
	} else if(opt->algorighm.find("nmf") !=std::string::npos) {
		model.init(opt->algorighm, nx, opt->algParam, unsigned(1)); // seed 1??
	} else if(opt->algorighm.find("lda") !=std::string::npos) {
		model.init(opt->algorighm, 10434, opt->algParam, unsigned(1)); // seed 1??
	} else
		model.init(opt->algorighm, nx, opt->algParam, 0.01);
}

void Master::sendParameter(const int target, const bool sendVersion)
{	
	std::vector<double> np;
	if (opt->algorighm == "lda"){
		np = model.getParameter().getLDAweights();
	} else if (opt->algorighm == "gmm"){
		np = model.getParameter().getGMMweights(K, nx, NCnt);
	} else {
		np = move(model.getParameter().getWeights());
	}

	// 	// np.push_back(nUpdate); //////////////// append param version
	// 	DVLOG(3) << "send parameter to " << target << " with: " << np.size();
	// 	net->send(wm.lid2nid(target), MType::DParameter, np);
	// 	std::vector<double> np = model.getParameter().getLDAweights();
	// 	np.push_back(nUpdate); //////////////// append param version
	// 	DVLOG(3) << "send parameter to " << target << " with: " << np.size();
	// 	net->send(wm.lid2nid(target), MType::DParameter, np);
	// } else {
	if (sendVersion)
		np.push_back(nUpdate); //////////////// append param version
	DVLOG(3) << "send parameter to " << target << " with: " << np.size()
			<< ", nUpdate: " << nUpdate;
	net->send(wm.lid2nid(target), MType::DParameter, np);
	if (sendVersion)
		np.pop_back();
	
	++stat.n_par_send;
}

void Master::broadcastParameter(const bool sendVersion)
{	
	VLOG_IF(revDelta.size() > 1 && iter<30, 1) << "rev Delta stat: " 
			<< revDelta.back()-revDelta.front() << "; " << revDelta << ", " << shrinkFactor;
	revDelta.clear();
	if (opt->algorighm == "lda"){
		std::vector<double> np = model.getParameter().getLDAweights();
		np.push_back(nUpdate); //////////////// append param version
		DVLOG(3) << "broad parameter: " << np.size() << " ; " << np;
		net->broadcast(MType::DParameter, np);
		// np.pop_back();
	} else {
		if (sendVersion)
			model.getParameter().weights.push_back(nUpdate); //////////////// append param version
		DVLOG(3) << "broad parameter: " << model.getParameter().weights.size() << " ; " << model.getParameter().weights;
		net->broadcast(MType::DParameter, model.getParameter().weights);
		if (sendVersion)
			model.getParameter().weights.pop_back();
	}
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
	// if(opt->algorighm == "km" || opt->algorighm == "mlp") {
	// 	return true;
	// }
	else if (opt->algorighm.find("nmf") !=std::string::npos
		|| opt->algorighm.find("lda") !=std::string::npos ) {
		if(iter < 8 || iter == 10 || iter == 14 || iter == 19
			|| iter >= lastArchIter * 3 / 2
			|| tmrArch.elapseSd() >= opt->arvTime)
		{
			lastArchIter = iter;
			tmrArch.restart();
			return true;
		}
	}
	else if(iter - lastArchIter >= opt->arvIter
		|| tmrArch.elapseSd() >= opt->arvTime)
	{
		lastArchIter = iter;
		tmrArch.restart();
		return true;
	}
	return false;
}

bool Master::archiveProgress(const bool force)
{
	if(!force && !needArchive())
		return false;
	foutput << iter << "," << tmrTrain.elapseSd();
	
	foutput << "," << ttDpProcessed << "," << objImproEsti;
			// << "_" << objEsti << "__" << staleStats;

	if (opt->algorighm.find("lda") !=std::string::npos){
		// VLOG(2) << "------ archive LDA beta";
		vector<double> beta = model.getParameter().getLDAweights();
		for(auto& v : beta){
			foutput << "," << v;
		}
	} else {
		// VLOG(2) << "------ archive param";
		for(auto& v : model.getParameter().weights){
			foutput << "," << v;
		}
	}
	foutput <<"\n";
	staleStats = "";
	objEsti = 0;
	objImproEsti = 0;
	return true;
}

bool Master::needArchiveAsync(int it)
{
	if(!foutput.is_open())
		return false;
	if(it < 10 // or (nUpdate < 10 and nUpdate % (2) == 0) 
		or (it < 21 and it % (2) == 0)
		or (it < 81 and it % (4) == 0) 
		or (it < 101 and it % (8) == 0) 
		or (it < 201 and it % (16) == 0) 
		or (it < 801 and it % (32) == 0) 
		or it % 100 == 0
		// or (nUpdate > 700 and nUpdate < 800)
		// or (nUpdate > 1700 and nUpdate < 1800)
		){
		lastArchIter = it;
		tmrArch.restart();
		return true;
	}
	return false;
}

bool Master::archiveProgressAsync(std::string staleStats, const bool force)
{
	// int iterCnt = (opt->mode.find("asyc") !=std::string::npos) ? nUpdate : iter;

	if(!force && !needArchiveAsync(nUpdate))
		return false;
	foutput << nUpdate << "," << tmrTrain.elapseSd();
	// if (ttDpProcessed > 0){
	foutput << "," << ttDpProcessed;
	// }
	if(staleStats.size() > 0)
		foutput << "," << staleStats;

	if (opt->algorighm.find("lda") !=std::string::npos){
		// VLOG(2) << "------ archive LDA beta";
		vector<double> beta = model.getParameter().getLDAweights();
		for(auto& v : beta){
			foutput << "," << v;
		}
	} else {
		for(auto& v : model.getParameter().weights){
			foutput << "," << v;
		}
	}
	foutput <<"\n";
	// foutput.flush();
	return true;
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
	// VLOG(2) << "receive online msg from " << info.source << "; " << lid;
	rph.input(MType::COnline, lid);
	sendReply(info);
}

void Master::handleXLength(const std::string& data, const RPCInfo& info){
	Timer tmr;
	int source = wm.nid2lid(info.source);

	if(opt->algorighm == "km" || opt->algorighm == "gmm") {
		// if(source == 0) { // initial parameter from top k data in worker 0
			std::vector<double> centroids = deserialize<std::vector<double>>(data);
			stat.t_data_deserial += tmr.elapseSd();
			// int k = stoi(opt->algParam);
			int numK = centroids.back();
			if(numK == 0) return;
			centroids.pop_back();
			if(nx == 0){
				nx = centroids.size()/numK;
			} else if(nx != centroids.size()/numK){
				LOG(FATAL)<<"dataset on "<<source<<" does not match with others: "<<
					nx << " " << numK << " " << centroids.size()/K;
			}
			if (candiParam.empty()){
				candiParam.assign(nWorker, vector<double>());
			}
			candiParam[source] = centroids;
			// VLOG(2) << "s: " << source << ": " << candiParam[source];

			// if (candiParam.empty() || (source == 0 && !candiParam.empty() && k == numK)) {
			// 	candiParam = move(centroids);
			// } else if (candiParam.size() < k * nx){
			// 	int remain = k - candiParam.size()/nx;
			// 	if(remain >= numK){
			// 		candiParam.insert(candiParam.end(), centroids.begin(), centroids.end());
			// 	} else{
			// 		candiParam.insert(candiParam.end(), centroids.begin(),
			// 			centroids.begin() + remain*nx);
			// 	}
			// }
		// }
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
	revDelta.push_back(tmrTrain.elapseSd());
	auto delta = deserialize<vector<double>>(data);
	stat.t_data_deserial += tmr.elapseSd();
	int s = wm.nid2lid(info.source);

	if (opt->algorighm == "km"){
		double obj = delta.back();
		delta.pop_back();
		double objImprove = delta.back();
		delta.pop_back();

		staleStats += "_" + std::to_string(s) + "_" + std::to_string(obj) 
			+ "_" + std::to_string(objImprove);
		objEsti += obj;
		objImproEsti += objImprove;
	}
	else if (opt->algorighm == "mlp"){
		double objImprove = l1norm0(delta);

		staleStats += "_" + std::to_string(s) + "_" + std::to_string(objImprove);
		objImproEsti += objImprove;
	}
	applyDelta(delta, s);
	ttDpProcessed += opt->batchSize/nWorker;
	rph.input(typeDDeltaAll, s);
	rph.input(typeDDeltaAny, s);
	//sendReply(info);
	++stat.n_dlt_recv;
}

void Master::handleDeltaAsync(const std::string & data, const RPCInfo & info)
{
	Timer tmr;
	int s = wm.nid2lid(info.source);
	auto delta = deserialize<vector<double>>(data);
	size_t staleParam = delta.back();
	delta.pop_back();
	curStale = nUpdate - staleParam;
	// if(lastArchIter != nUpdate and curStale > 20){
	// 	archiveProgressAsync(curStats, true);
	// }
	curStats = std::to_string(nUpdate) + "--" + std::to_string(staleParam) 
			+ "--" + std::to_string(curStale) + "--" + std::to_string(s);

	staleStats += curStats + "\n";

	stat.t_data_deserial += tmr.elapseSd();
	applyDelta(delta, s);
	ttDpProcessed += opt->batchSize/nWorker;
	++nUpdate;
	// if(curStale > 20){
	// 	archiveProgressAsync(curStats, true);
	// }
	// else{
		archiveProgressAsync(curStats);
	// }
	rph.input(typeDDeltaAll, s);
	rph.input(typeDDeltaAny, s);
	//sendReply(info);
	++stat.n_dlt_recv;
	// directly send new parameter
	sendParameter(s, true);
}

void Master::handleDeltaProgAsync(const std::string & data, const RPCInfo & info)
{
	Timer tmr;
	int s = wm.nid2lid(info.source);
	auto delta = deserialize<vector<double>>(data);
	int deltaCnt = delta.back();
	getDeltaCnt += deltaCnt;
	delta.pop_back();

	deltaCount[s] += deltaCnt;
	deltaObj[s] += delta.back();
	objEsti += delta.back();
	delta.pop_back();
	deltaT[s] += tmrDeltaV.elapseSd();
	objImproEsti += delta.back();

	stat.t_data_deserial += tmr.elapseSd();
	applyDelta(delta, s);

	double thread = glbBatchSize;
	if(opt->mode.find("pasp4") !=std::string::npos){
		thread = glbBatchSize * shrinkFactor;
	}
	//// pasp5 send out the interval
	if(opt->mode.find("pasp5") !=std::string::npos && nUpdate == 0){
		double tt = tmrTrain.elapseSd();
		VLOG(1) << "Broadcast Interval: " << tt << " for " << deltaCnt << " from " << s;
		net->broadcast(MType::CTrainInterval, tt * opt->reptr / deltaCnt / nWorker);
	}
	
	if (opt->algorighm == "mlp"){
		double objImprove = l1norm0(delta);
		staleStats += "_" + std::to_string(s) + "_" + std::to_string(objImprove);
		objImproEsti += objImprove;
	}

	VLOG(3) << " Rev Delta from " << s << ", " << deltaCnt << ", getDeltaCnt: " << getDeltaCnt 
			<< ", thread: " << thread;

	rph.input(typeDDeltaAll, s);

	if (getDeltaCnt >= thread) { // && opt->mode.find("pasp5") == std::string::npos) {
		rph.input(typeDDeltaAny, s);
		// ++unSendDelta;
	// if (unSendDelta >= freqSendParam){
		broadcastParameter();
		sentDReq = false;
		tmrDeltaV.restart();
		deltaV.assign(nWorker+3, 0);
		ttDpProcessed += getDeltaCnt;
		reportNum = 0;
		fastReady = false;
		factorReady = false;
		
		++nUpdate;
		// VLOG_IF(nUpdate<5,1) << "pasp4 shrinkFactor: " << shrinkFactor;
		string states = std::to_string(getDeltaCnt) + ";" + std::to_string(objEsti) 
			+ ";" + std::to_string(objImproEsti);
		for (int i = 0; i < nWorker; i++){
			states += ";" + std::to_string(deltaCount[i])+"_"+std::to_string(deltaObj[i])+"_"
					+std::to_string(deltaT[i]);
		}

		deltaCount.assign(nWorker, 0);
		deltaT.assign(nWorker, 0.0);
		deltaObj.assign(nWorker, 0.0);

		// archiveProgressAsync(std::to_string(shrinkFactor)+"_"
		// 		+std::to_string(objImproEsti/getDeltaCnt), false);
		archiveProgressAsync(states, false);
		shrinkFactor = 1.0;
		getDeltaCnt = 0;
		objImproEsti = 0.0;
		objEsti = 0.0;
	// }
	}
	//sendReply(info);<< "," << objImproEsti
	++stat.n_dlt_recv;
	// directly send new parameter
	// sendParameter(s);
	// ++unSendDelta;
	// if (unSendDelta >= freqSendParam){
	// 	broadcastParameter();
	// 	++nUpdate;
	// 	unSendDelta = 0;
	// 	archiveProgressAsync("", true);
	// }
}
void Master::handleReport(const std::string & data, const RPCInfo & info)
{
	Timer tmr;
	revDelta.push_back(tmrTrain.elapseSd());
	if (sentDReq) //// ????
		return;

	auto rpt = deserialize<std::vector<double> >(data);
	reportCnt += rpt[0];
	reportNum++;
	stat.t_data_deserial += tmr.elapseSd();
	int s = wm.nid2lid(info.source);
    VLOG_IF(nUpdate < 20, 3) << "Delta report " << s << ", " << deltaV;

	double cutoff = glbBatchSize;
	//// pasp4 shrink the glbBatchSize
	if(opt->mode.find("pasp4") !=std::string::npos){

		//// decide normal workers and stragglers
		if (deltaV[s] == 0){
			deltaV[s] = 1/tmrDeltaV.elapseSd();
			deltaV[nWorker+1] = (deltaV[nWorker+1]*deltaV[nWorker] + deltaV[s])/(deltaV[nWorker]+1);
			deltaV[nWorker] += 1;
			if (!fastReady)
				deltaV[nWorker+2] = deltaV[s];
		}
		if (!fastReady && ( (int)deltaV[nWorker]*2 == nWorker) || reportNum > nWorker){
			fastReady = true;
		}
		if (fastReady && !factorReady){
			int cnt = 0;
			double sum = 0;
			for (int i = 0; i < nWorker; i++){
				if (deltaV[i] > deltaV[nWorker+2] * 0.9){
					sum += deltaV[i];
					cnt += 1;
				}
			}
			shrinkFactor = deltaV[nWorker+1]*deltaV[nWorker]/nWorker * cnt/sum;
			if ((int)deltaV[nWorker] == nWorker) 
				factorReady = true;
			VLOG_IF(nUpdate < 9 && factorReady, 1) << "V report " << deltaV 
					<< ", " << deltaV[nWorker+2] << ", " << shrinkFactor;
		}
		cutoff = glbBatchSize * shrinkFactor;
	}
	//// pasp5 measure obj score
	else if (opt->mode.find("pasp5") !=std::string::npos){
		

	}

	if (reportCnt >= cutoff){
		net->broadcast(MType::DDeltaReq, "");
		sentDReq = true;
		reportCnt = 0;
	}
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
	if(opt->algorighm != "km" && opt->algorighm.find("nmf")!=std::string::npos) {
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
	// VLOG(2) << "Master receive parameter: ";
	auto weights = deserialize<vector<double>>(data);
	// VLOG(2) << "receive parameter: " << weights.size() << "; " << model.paramWidth();
	Parameter p;
	p.set(move(weights));
	// VLOG(3) << "apply parameter: " << nIterChange << "; " << p.weights;
	checkParamChange(p);
	model.setParameter(p);
	suParam.notify();
	//sendReply(info);
	++stat.n_par_recv;
}

void Master::checkParamChange(const Parameter& p){
	
	// if(iter == 100) {
	// 	VLOG(1) << "model parameter: " << model.getParameter().weights;
	// 	VLOG(1) << "master parameter: " << param.weights;
	// 	VLOG(1) << "new parameter: " << p.weights;
	// }
	if(model.getParameter().isSameParm(p)) {
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
