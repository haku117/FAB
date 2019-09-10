#include "Worker.h"
#include "common/Option.h"
#include "network/NetworkThread.h"
#include "message/MType.h"
#include "logging/logging.h"
#include "train/EM.h"
#include "train/GD.h"
#include "util/Util.h"
#include "math.h"
#include <unistd.h>

using namespace std;

///////---- Basic function start ----/////////
// initialize constant / stable data
Worker::Worker() : Runner() {
	masterNID = 0;
	dataPointer = 0;
	iter = 0;
	localBatchSize = 1;
	forceCalT = 0;

	hasNewParam = false;
	allowTrain = true;
	exitTrain = false;

	/// get from master
	typeDDeltaAny = MType::DDelta;
	typeDDeltaAll = 128 + MType::DDelta;
	//workerLst = {}; // ??
	//trainer->bindModel(&model);
	factorDelta = 1.0;
	nx = 0;
	nUpdate = 0;
	lastArchIter = 0;
	bfDeltaCnt = 0;
	bfDeltaCntExt = 0;
	isbfDeltaExt = false;
	t_wdelta = 0;

	suOnline.reset();
	suParam.reset();
	suXlength.reset();

	/// for DC
	// suDeltaAny.reset();
	suDeltaAll.reset();
	// suTPause.reset();
}
void Worker::init(const Option* opt, const size_t lid) {
	this->opt = opt;
	nWorker = opt->nw;
	localID = lid;

	ln = opt->logIter;
	logName = "W"+to_string(localID);
	setLogThreadName(logName);

	std::vector<int> tokens = parseParam(opt->straggler, ",");
	int tn = tokens.size();
	lamda = tokens[0];
	range = lamda > 0 && tn > 1 ? tokens[1] : 0;
	if(lamda > 0) {
		gen = mt19937(localID);
		distribution = exponential_distribution<double>(lamda);
	}

	reportSize = opt->reptr;
	reqDelta = false;
	hasNewParam = false;
	interval = 99999.9;

	// int indx = opt->algParam.rfind(',');
	// if (indx > 0)
	// 	delayWorkers = stoi(opt->algParam.substr(indx+1));
	// 	if (delayWorkers > 0)
	// 		forceCalT = -1;
	// else 
	// 	delayWorkers = -1;
	// delay option
	// if (opt->mode.length() > 4) {
	// 	if (localID < delayWorkers){
	// 		interval = stoi(opt->mode.substr(4));
	// 	}
	// }

	if (opt->algorighm.find("km") !=std::string::npos) {
		trainer = new EM;
		// if (tn > 1){
		// 	lamda = double(tokens[1]);
		// 	range = tokens[2];
		// 	// VLOG(1) << "new trainer " << opt->algorighm << ", lamda: " << lamda 
		// 	// 		<< ", range: " << range;
		// }
	}else if (opt->algorighm.find("nmf") !=std::string::npos) {
		trainer = new EM;
		trainer->setRate(opt->lrate);
	}else if (opt->algorighm.find("lda") !=std::string::npos 
		|| opt->algorighm.find("gmm") !=std::string::npos) {
		trainer = new EM;
		// trainer->setRate(opt->lrate); //?? need?
	}else {
		trainer = new GD;
		trainer->setRate(opt->lrate);
		// if (opt->algorighm == "mlp" && tn > 4){
		// 	lamda = double(tokens[tn-3]);
		// 	range = tokens[tn-2];
		// }
		// if (opt->algorighm == "lr"){
		// 	lamda = double(tokens[1]);
		// 	range = tokens[2];
		// }
		// VLOG_IF(localID == 0, 1) << "new GD trainer "<< opt->algorighm << ", lamda: " << lamda 
		// 			<< ", range: " << range << ", tok: " << tokens;
	}
	/// initial delayArr
	// for (int i = 0; i < range; i++){
	// 	delayArr.push_back(lamda * exp(-lamda * i) * 10);
	// }
	VLOG_IF(localID == 0, 1) << "DelayFactor for " << lamda << ", " << range << ", " << reportSize;
	// srand(localID);

	/// for dc cache
	deltaIndx0.assign(nWorker, false);
	deltaIndx1.assign(nWorker, false);
	deltaReceiver.assign(nWorker, false);
	tmrGlb.restart();
	curCalT = 0;
	curCnt = 0;
	deltaWaitT = std::vector<double>();
	curHlvl = 0;
	mylvl = localID == 0? nWorker :id2lvl(localID);
	dstgrpID = dstGrpID(localID, mylvl);

	/// multicast
	mltDD = 3; /// multicast degree

	if(opt->mode.find("sync")!=std::string::npos){
		syncInit();
	} else if(opt->mode.find("asyc")!=std::string::npos){
		asyncInit();
	} else if(opt->mode.find("pasp") !=std::string::npos){
		// if (int p = opt->mode.find('-')) {
		// 	VLOG_IF(localID == 0, 1) << "report "<< opt->mode << ", " << p;
		// 	// reportSize = stoi(opt->mode.substr(p+1));
		// }
		progAsyncInit();
	} else if(opt->mode == "fsb"){
		fsbInit();
	} else if(opt->mode == "fab"){
		fabInit();
	} else if(opt->mode.find("dc") !=std::string::npos){
		dcSyncInit();
	} else if(opt->mode.find("pipe") !=std::string::npos){
		DLOG_IF(localID < 4, INFO) << "pipe Initalization";
		pipeInit();
	}
}
void Worker::run() {
	LOG_IF(localID < 4, INFO) << "register handlers";
	registerHandlers();  // register message with function, function need to wrap up
	startMsgLoop(logName+"-MSG"); // make a new thread to record messages

	LOG_IF(localID < 4, INFO) << "start";
	DLOG_IF(localID < 4, INFO) << "send online message";
	sendOnline();

	DLOG_IF(localID < 4, INFO) << "waiting worker list";
	waitWorkerList();
	DLOG_IF(localID < 4, INFO) << "send x length " << trainer->pd->xlength();
	sendXLength(); // x dimension
	DLOG_IF(localID < 4, INFO) << "waiting init parameter";
	waitParameter();
	DLOG(INFO) << "got init parameter: ";
	DLOG_IF(localID == 0, INFO) << bfParam.weights.size() << ", " << bfParam.weights;
	model.init(opt->algorighm, trainer->pd->xlength(), opt->algParam);
	VLOG(3) << "finish init model";
	trainer->bindModel(&model); /// move
	VLOG(3) << "finish bind model";
	trainer->initState(1);
	VLOG(3) << "finish init state";
	applyBufferParameter();
	resumeTrain();

	DLOG_IF(localID < 4, INFO) << "start training with mode: " << opt->mode 
			<< ", local batch size: " << localBatchSize << ", report size: " << reportSize;
	iter = 0;

	int indx = opt->mode.find(",");
	if(indx != std::string::npos) {
		string strStale = opt->mode.substr(indx+1);
		VLOG(1) << "staleness " << strStale;
		staleness = stoi(strStale);
	}

	//try{
	//	generalProcess();
	//} catch(exception& e){
	//	LOG(FATAL) << e.what();
	//}
	if(opt->mode.find("sync") !=std::string::npos){
		syncProcess();
	} else if(opt->mode.find("asyc") !=std::string::npos){
		asyncProcess();
	} else if(opt->mode.find("pasp") !=std::string::npos){
		progAsyncProcess();
	} else if(opt->mode == "fsb"){
		fsbProcess();
	} else if(opt->mode == "fab"){
		fabProcess();
	// } else if(opt->mode == "dcsync"){
	// 	dcSyncProcess();
	} else if(opt->mode.find("dc") !=std::string::npos){
		LOG_IF(localID < 4, INFO) << "dcFsb Process ";
		dcFsbProcess();
	} else if(opt->mode.find("pipe") !=std::string::npos){
		LOG(INFO) << "pipe Process" ;
		pipeProcess();
	}

	DLOG(INFO) << "finish training";
	sendClosed();
	finishStat();
	showStat();
	stopMsgLoop();
}
//\\\\\\ Basic function end \\\\\\\\//


///////---- DeCentralizated Model function start ----/////////
void Worker::dcSyncInit()
{
	factorDelta = 1.0;// / nWorker;
	regDSPProcess(MType::DParameter, localCBBinder(&Worker::handleParameter));

	if(opt->mode.find("dcfsb") !=std::string::npos)
		regDSPProcess(MType::DDelta, localCBBinder(&Worker::handleDelta)); /// handle delta
	else if(opt->mode.find("dcring") !=std::string::npos)
		regDSPProcess(MType::DDelta, localCBBinder(&Worker::handleDeltaRingcast)); /// handle delta
	else if(opt->mode.find("dcmlt") !=std::string::npos)
		regDSPProcess(MType::DDelta, localCBBinder(&Worker::handleDeltaMltcast)); /// handle delta
	else if(opt->mode.find("grp") !=std::string::npos) {
		regDSPProcess(MType::DDelta, localCBBinder(&Worker::handleDeltaGrpcast)); /// handle delta
		regDSPProcess(MType::DDeltaRPL, localCBBinder(&Worker::handleDeltaRPL));
	} else if(opt->mode.find("dc1c") !=std::string::npos) {
		regDSPProcess(MType::DDelta, localCBBinder(&Worker::handleDelta)); /// handle delta
		regDSPProcess(MType::DDeltaRPL, localCBBinder(&Worker::handleDeltaRPLone));
	} else if(opt->mode.find("trans") !=std::string::npos) {
		regDSPProcess(MType::DDelta, localCBBinder(&Worker::handleDelta)); /// handle delta
		regDSPProcess(MType::DDeltaRPL, localCBBinder(&Worker::handleDeltaRPLtrans));
	} else if(opt->mode.find("dc2c") !=std::string::npos) {
		regDSPProcess(MType::DDelta, localCBBinder(&Worker::handleDelta2c)); /// handle delta
		regDSPProcess(MType::DDeltaRPL, localCBBinder(&Worker::handleDeltaRPL));
	}
	// addRPHAnySU(typeDDeltaAny, suDeltaAny);
	addRPHEachSU(typeDDeltaAll, suDeltaAll);
}

void Worker::pipeInit()
{
	factorDelta = 1.0;// / nWorker;
	initPipeBlk();
	regDSPProcess(MType::DParameter, localCBBinder(&Worker::handleParameter));
	// regDSPProcess(MType::CTrainSync, localCBBinder(&Worker::handleSync));

	if(opt->mode.find("fsb") !=std::string::npos)
		regDSPProcess(MType::DDelta, localCBBinder(&Worker::handleDeltaPipe)); /// handle delta
	else if(opt->mode.find("ring") !=std::string::npos)
		regDSPProcess(MType::DDelta, localCBBinder(&Worker::handleDeltaRingcast)); /// handle delta
	else if(opt->mode.find("mlt") !=std::string::npos)
		regDSPProcess(MType::DDelta, localCBBinder(&Worker::handleDeltaMltcast)); /// handle delta

	// addRPHAnySU(typeDDeltaAny, suDeltaAny);
	addRPHEachSU(typeDDeltaAll, suDeltaAll);
}

void Worker::dcFsbProcess()
{	
	while(!exitTrain){
		VLOG_EVERY_N(ln, 1) << "Iteration " << iter << ": calculate delta from " << dataPointer;
		Timer tmr;
		size_t cnt;
		
		std::vector<double> lclDelta;
		// try to use localBatchSize data-points, the actual usage is returned via cnt 
		tie(cnt, lclDelta) = trainer->batchDelta(allowTrain, dataPointer, localBatchSize, true);
		curCalT = tmr.elapseSd();
		// VLOG(2) << " calculate time: " << curCalT << " for dp: " << cnt;
		updatePointer(cnt);
		double updateDpT = tmr.elapseSd() - curCalT;
		VLOG_IF(iter<2 && (localID < 9), 1) << "unit dp cal time for " << cnt << " : " << curCalT/cnt;
		VLOG_EVERY_N(ln, 3) << "  calculate delta with " << cnt << " data points";
		// VLOG_EVERY_N(ln/10, 2) << "  current iter cal time: " << tmr.elapseSd();
		stat.t_dlt_calc+= tmr.elapseSd();
		// VLOG_EVERY_N(ln, 3) << "  send delta";

		tmr.restart();
		if(allowTrain == true) { broadcastSignalPause();}
		
		if(opt->mode.find("dcfsb") !=std::string::npos)
			broadcastDelta(lclDelta);
		else if(opt->mode.find("dcring") !=std::string::npos)
			ringcastDelta(lclDelta);
		else if(opt->mode.find("dcmlt") !=std::string::npos)
			multicastDelta(lclDelta);
		else if(opt->mode.find("dc1c") !=std::string::npos)
			singlecastDelta(lclDelta);
		else if(opt->mode.find("dc2c") !=std::string::npos)
			dblecastDelta(lclDelta);
		else if(opt->mode.find("dctrans") !=std::string::npos)
			singlecastDelta(lclDelta);
		stat.t_dlt_send += tmr.elapseSd();

		double offset = tmrGlb.elapseSd(); // recompute the early received delta
		for(double tt : deltaWaitT){
			tt -= offset;
		}
		
		tmr.restart();
		tmrGlb.restart(); // for monitoring the delta ariving time
		if(opt->mode.find("grp") !=std::string::npos) {
			accumulateDelta(lclDelta, (int)localID, 0, iter);
			grpcastDelta();
		} 
		else {
			accumulateDelta(lclDelta, (int)localID);
		}
		stat.t_dlt_accumLcl += tmr.elapseSd();

		if(exitTrain){ break; }
		// VLOG_EVERY_N(ln, 2) << "  DC: wait for delta from all other workers";
		tmr.restart();
		waitDeltaFromAll();
		double wT = tmr.elapseSd();
		stat.t_par_wait += tmr.elapseSd();
		if(exitTrain){ break; }

		tmr.restart();

		/// send out final delta
		if(opt->mode.find("dc1c") !=std::string::npos && localID == 0){
			for(int ss = 1; ss < nWorker; ss++)
				net->send(wm.lid2nid(ss), MType::DDeltaRPL, bufferDelta);
		}
		else if(opt->mode.find("dctrans") !=std::string::npos && localID == 0)
			net->send(wm.lid2nid(1), MType::DDeltaRPL, bufferDelta);

		applyDelta();
		resumeTrain();
		if(localID == 0)	/// send record to master only for worker 0
			sendParameter2M(); /// update parameter to master
		stat.t_par_calc += tmr.elapseSd();

		deltaReceiver.assign(nWorker, false);
		curHlvl = 0;
		++iter;
	}
}

void Worker::pipeProcess()
{	
	Timer tmr;
	while(!exitTrain){
		VLOG_EVERY_N(ln, 1) << "Iteration " << iter << ": calculate delta ";
		tmr.restart();
		size_t cnt;
		int blk = iter % blkNum;
		
		std::vector<double> lclDelta;
		// try to use localBatchSize data-points, the actual usage is returned via cnt
		tie(cnt, lclDelta) = trainer->batchDeltaPipe(allowTrain, blkPointer[blk], 
			localBatchSize, blk, blkSize, true);
		updatePointerPipe(cnt, blk);
		/// chech the dp cal time wont be affected by the number of workers
		VLOG_IF(iter<2, 1) << "unit dp cal time for " << cnt << " : " << tmr.elapseSd()/cnt;
		VLOG_EVERY_N(ln, 3) << "  calculate delta with " << cnt << " data points in T:" 
			<< tmr.elapseSd();
		stat.t_dlt_calc+= tmr.elapseSd();

		tmr.restart();
		if(allowTrain) broadcastSignalPause(); // first worker done this iter
		
		// VLOG_EVERY_N(ln, 1) << "  send delta";
		if(opt->mode.find("fsb") !=std::string::npos)
			broadcastDeltaPlus(lclDelta);
		else if(opt->mode.find("ring") !=std::string::npos)
			ringcastDelta(lclDelta);
		else if(opt->mode.find("mlt") !=std::string::npos)
			multicastDelta(lclDelta);
		else if(opt->mode.find("hrky") !=std::string::npos)
			hrkycastDelta(lclDelta);

		accumulateDeltaPipe(lclDelta, (int)localID, iter);
		stat.t_dlt_accumLcl+= tmr.elapseSd();
		VLOG_EVERY_N(ln, 3) << "  applyed local delta in T:" << tmr.elapseSd();

		if(exitTrain) break;
		tmr.restart();
		int acnt = 0;
		while(iter >= stale + blkNum-1 && !exitTrain) {
			if(acnt++ % 100 == 0) {
				VLOG(2) << "Sleep at iter: " << iter << " stale: " << stale 
				<< " diff: " << (iter - stale) << " blkNum:" << blkNum; 
			}
			usleep(100);
		}
		stat.t_par_wait += tmr.elapseSd();

		resumeTrain();
		++iter;
	}
}
//\\\\\\ DeCentralizated Model function end \\\\\\\\//


///////---- initialization function start ----/////////
void Worker::bindDataset(const DataHolder* pdh){
	VLOG(1) << "Bind dataset with " << pdh->size() << " data points";
	trainer->bindDataset(pdh);
	VLOG(1) << "finish Bind dataset";
	// separated the mini-batch among all workers
	// if(opt->mode == "asyc")
	// 	localBatchSize = opt->batchSize;
	// else
		localBatchSize = opt->batchSize / nWorker;
	if(opt->batchSize % nWorker > localID)
		++localBatchSize;
	if(localBatchSize <= 0)
		localBatchSize = 1;
}

void Worker::initPipeBlk() {
	/// for pipe
	blkNum = 3; /// 3 level pipeline
	size_t indx;
	if 	( opt->algorighm.find("nmf") !=std::string::npos
		&& (indx = opt->algorighm.find(",")) != std::string::npos ){
			blkNum = stoi(opt->algorighm.substr(indx+1));
		} 

	stale = 0;

	// parse input param for nnx and nny
	std::vector<int> tokens = parseParam(opt->algParam);
	nny = tokens[2];
	int uny = nny / blkNum;

	// initialize update pointers
	blkPointer.assign(blkNum, 0);
	blkSize.assign(blkNum, uny);
	for(int i = 1; i < blkNum; i++){
		blkPointer[i] = uny * i;
	}
	blkSize[blkNum-1] = nny - uny * (blkNum-1);
	// VLOG(2) << "initialize pipe blk " << tokens << " nny: " << nny << " uny: " << uny
	// 	<< " blkPointer: " << blkPointer << " blkSize: " << blkSize;

	// initialize delta buffer
	curDeltaIndex = 0;
	bfBlkDelta.assign(blkNum, std::vector<double>());
	bfBlkDeltaIndex.assign(blkNum, std::vector<bool>(nWorker, false));
	blkDeltaBFCnt.assign(blkNum, 0);
}

void Worker::sendOnline(){
	net->send(masterNID, MType::COnline, localID);
}
void Worker::waitWorkerList(){
	suOnline.wait();
}
void Worker::sendXLength(){
	if(opt->algorighm == "km" || opt->algorighm == "gmm") {
		/// send k dp as candidate of global centroids
		std::vector<double> kCentroids;
		int k = 0;
		int indx = opt->algParam.find(',');
		if (indx < 0)
			k = stoi(opt->algParam);
		else
			k = stoi(opt->algParam.substr(0, indx));
		int i = 0;
		int avgk = k / nWorker;
		if (localID < k % nWorker)
			avgk += 1;
		// for(; i < k && i < trainer->pd->size(); i++){
		for(; i < avgk; i++){
			// int dp = rand() % trainer->pd->size(); /// random seed......
			std::vector<double> OneDp = trainer->pd->get(i).x;
			kCentroids.insert(kCentroids.end(), OneDp.begin(), OneDp.end());
			if(opt->algorighm == "km")
				kCentroids.push_back(1); // for cluster counts
		}
		VLOG(2) << "s " << localID << ", " << i << ": " << kCentroids;
		kCentroids.push_back(i);
		net->send(masterNID, MType::CXLength, kCentroids);
	}
	// else if(opt->algorighm == "lda") {
	// 	net->send(masterNID, MType::CXLength, num_terms);
	// }
	else
		net->send(masterNID, MType::CXLength, trainer->pd->xlength());
	suXlength.wait();
}
void Worker::sendClosed(){
	net->send(masterNID, MType::CClosed, localID);
}
void Worker::broadcastSignalPause() { net->broadcast(MType::CTrainPause, ""); }
//\\\\\\ initialization function end \\\\\\\\//



///////---- Delta Process function start ----/////////
//////===== Send delta
void Worker::sendDelta(std::vector<double>& delta, const int ss)
{
	// TODO: add staleness tolerance logic here
	// ss for paramVersion for async or update count for pasp

	//DVLOG_EVERY_N(ln, 1) << "n-send: " << iter << " un-cmt msg: " << net->pending_pkgs() << " cmt msg: " << net->stat_send_pkg;
	// if (opt->algorighm == "lda" || opt->algorighm == "gmm"){
	// 	std::vector<double> dd = model.computeDelta(delta);
	// 	DVLOG(3) << "send dd " << dd.size() << " : " << dd;
	// 	net->send(masterNID, MType::DDelta, dd);
	// }
	// else {
		int size = delta.size();
		if (opt->algorighm == "km") {
			// delta[size-1] /= localBatchSize;
			// delta[size-2] /= localBatchSize;
			DVLOG(2) << "send delta: " << delta[size-1] << "; " << delta[size-2];
		}
		if (ss > 0)
			delta.push_back(ss); /////// paramVersion
		DVLOG(3) << "; delta: " << ss << "; " << size << "; " << delta;
		net->send(masterNID, MType::DDelta, delta);
		if (ss > 0)
			delta.pop_back();
	// }
	++stat.n_dlt_send;
}

void Worker::sendReport(std::vector<double>& report)
{
	DVLOG(3) << "send report: " << report;
	net->send(masterNID, MType::DReport, report);
}

void Worker::broadcastDelta(std::vector<double>& delta)
{
	// TODO: add staleness tolerance logic here
	DVLOG(3) << "broadcast delta: " << delta;
	//DVLOG_EVERY_N(ln, 1) << "n-send: " << iter << " un-cmt msg: " << net->pending_pkgs() << " cmt msg: " << net->stat_send_pkg;
	///// for(int lid; lid < )
	net->broadcast(MType::DDelta, delta);
	// for(int i = 0; i < nWorker; i++){
	// 	if(i != localID){
	// 		net->send(wm.lid2nid(i), MType::DDelta, delta);
	// 	}
	// }
	++stat.n_dlt_send;
}
void Worker::broadcastDeltaPlus(std::vector<double>& delta)
{
	delta.push_back(localID); // add original source
	delta.push_back(iter); // add delta iter #
	DVLOG(3) << "broadcast delta: " << delta;
	// net->broadcast(MType::DDelta, delta);
	for(int i = 0; i < nWorker; i++){
		if(i != localID){
			net->send(wm.lid2nid(i), MType::DDelta, delta);
		}
	}
	++stat.n_dlt_send;
}
void Worker::ringcastDelta(std::vector<double>& delta)
{
	// TODO: add staleness tolerance logic here
	//DVLOG_EVERY_N(ln, 1) << "n-send: " << iter << " un-cmt msg: " << net->pending_pkgs() << " cmt msg: " << net->stat_send_pkg;
	///// for(int lid; lid < )
	delta.push_back(localID); // add original source
	delta.push_back(iter); // add delta iter #

	int nbl = (localID -1 + nWorker) % nWorker;
	int nbr = (localID +1 + nWorker) % nWorker;
	DVLOG(3) << "ringcast delta: " << nbl << "," << nbr << ", " << delta;
	net->send(wm.lid2nid(nbl), MType::DDelta, delta);
	if (nbl != nbr)
		net->send(wm.lid2nid(nbr), MType::DDelta, delta);
	++stat.n_dlt_send;
}
void Worker::multicastDelta(std::vector<double>& delta)
{
	// TODO: add staleness tolerance logic here
	//DVLOG_EVERY_N(ln, 1) << "n-send: " << iter << " un-cmt msg: " << net->pending_pkgs() << " cmt msg: " << net->stat_send_pkg;
	///// for(int lid; lid < )
	delta.push_back(localID); // add original source
	delta.push_back(iter); // add delta iter #

	std::set<int> nbs;
	while(nbs.size() < mltDD && nbs.size() < (nWorker-1)){
		int nb = rand() % nWorker;
		if(nb != localID && nbs.find(nb) != nbs.end()) {
			nbs.insert(nb);
		}
	}
 
	DVLOG(3) << "ringcast delta: " << nbs.size() << ", " << delta;
	for (int nb : nbs){
		net->send(wm.lid2nid(nb), MType::DDelta, delta);
	}
	++stat.n_dlt_send;
}
void Worker::hrkycastDelta(std::vector<double>& delta)
{
	delta.push_back(0); // add hierarchy level
	delta.push_back(iter); // add delta iter #
	delta.push_back(localID); // add original source
	delta.push_back(1); // add delta count
	accuDelta.push_back(localID); // local accumulated delta
	
	int nb = localID % 2 == 0 ? localID + 1 : localID - 1;
	if (nb < nWorker){
		net->send(wm.lid2nid(nb), MType::DDelta, delta);
	}
	++stat.n_dlt_send;
}
void Worker::grpcastDelta()
{	
	if(localID!=0 /// final group leader 
		&& (mylvl == 0 // lvl 0 workers, localID == odd
			|| localID + 1 >= nWorker // single even last worker
			|| bfDeltaCnt == int(pow(2, mylvl)) ) ){ // required delta has already been received

		// VLOG_IF(bfDeltaCnt > 0 , 1) << " ****** send orig adv delta: " << bfDeltaCnt;
		bufferDelta.push_back(iter); // add iter #
		net->send(wm.lid2nid(dstgrpID), MType::DDelta, bufferDelta);
		++stat.n_dlt_send;
		// VLOG(2) << "delta sent to " << dstgrpID << " w hlvl: " << mylvl;
	}
}

void Worker::singlecastDelta(std::vector<double>& delta){
	if (localID != 0){
		// delta.push_back(iter); // add iter #
		VLOG(2) << " send delta to w0";
		net->send(wm.lid2nid(0), MType::DDelta, delta);
		++stat.n_dlt_send;
	}
}
void Worker::dblecastDelta(std::vector<double>& delta){
	if (localID > 1){
		// delta.push_back(iter); // add iter #
		net->send(wm.lid2nid(localID % 2), MType::DDelta, delta);
		++stat.n_dlt_send;
	} else if (nWorker == 2){
		net->send(wm.lid2nid(1 - localID), MType::DDelta, delta);
	}
}


//////====== Process delta
void Worker::waitDeltaFromAll(){
	suDeltaAll.wait();
	suDeltaAll.reset();
}

void Worker::accumulateDelta(const std::vector<double>& delta)
{
	for (size_t i = 0; i < delta.size(); ++i)
		bfDelta[i] += delta[i];
}
void Worker::accumulateDelta(std::vector<double>& delta, const int source)
{
	lock_guard<mutex> lk(mDelta);
	if (deltaIndx0[source]) { // if a delta from source is already there
		copyDelta(bufferDeltaExt, delta);
		deltaIndx1[source] = true;
		bfDeltaCntExt++;
	}
	else {
		DVLOG_IF(deltaIndx1[source], 1) << "xxxxxxxx Dam WWWTTTFFFF number of delta applied &&&&&&&";
		copyDelta(bufferDelta, delta);
		deltaIndx0[source] = true;
		bfDeltaCnt++;
		rph.input(typeDDeltaAll, source); // trigger the syncUnit counter
	}
}
void Worker::accumulateDelta(std::vector<double>& delta, const int source, const size_t hlvl, const size_t diter)
{
	lock_guard<mutex> lk(mDelta);
	int powhlvl = pow(2, hlvl);
	int newcnt = powhlvl + source > nWorker ? nWorker - source : powhlvl;
	int i = 0;

	// if (deltaIndx0[source]) { // if a delta from source is already there
	// 	// DVLOG(1) << "|||||||| process advanced delta from s: " << source << " indx: " << deltaIndx0;
	// 	copyDelta(bufferDeltaExt, delta);
	// 	for (; i < powhlvl && source+i < nWorker; i++) {
	// 		deltaIndx1[source + i] = true;
	// 	}
	// 	bfDeltaCntExt += newcnt;
	// }
	// else {
		DVLOG_IF(deltaIndx0[source], 1) << "xxxxxxx process advanced delta from s: " << source << " indx: " << deltaIndx0;
		copyDelta(bufferDelta, delta);
		bfDeltaCnt += newcnt;
			
		if(bfDeltaCnt == nWorker) {  /// for grp cast
			deltaWaitT.push_back(tmrGlb.elapseSd());
			// VLOG(2) << "broadcast rpl delta from " << localID;
			for(int src : recSrcs) {
				net->send(wm.lid2nid(src), MType::DDeltaRPL, bufferDelta);
			}
			if (nWorker > 3)
				net->send(wm.lid2nid(3), MType::DDeltaRPL, bufferDelta);
			recSrcs.clear();
			// net->broadcast(MType::DDeltaRPL, bufferDelta);
			deltaWaitT.push_back(tmrGlb.elapseSd());
		}
		for ( ; i < powhlvl && source+i < nWorker; i++) {
			deltaIndx0[source + i] = true;
			rph.input(typeDDeltaAll, source+i); // trigger the syncUnit counter
		}
	// }
}
void Worker::accumulateDelta(std::vector<double>& delta, const std::vector<int>& sources)
{
	lock_guard<mutex> lk(mDelta);
	if (deltaIndx0[sources[0]]) { // if a delta from source is already there
		copyDelta(bufferDeltaExt, delta);
		// deltaIndx1[sources[0]] = true;
		for (int ss : sources){
			deltaIndx1[ss] = true;
		}
		accuDeltaExt.insert(accuDeltaExt.end(), sources.begin(), sources.end());
		bfDeltaCnt += sources.size();
		// DVLOG_IF(bfDeltaCnt > nWorker, 2) << " Dam MORE number of delta applied &&&&&&&";
	}
	else {
		// DVLOG_IF(deltaIndx1[source], 2) << " Dam WWWTTTFFFF number of delta applied &&&&&&&";
		copyDelta(bufferDelta, delta);
		for (int ss : sources){
			deltaIndx0[ss] = true;
			rph.input(typeDDeltaAll, ss); // trigger the syncUnit counter
		}
		accuDelta.insert(accuDelta.end(), sources.begin(), sources.end());
	}
}
void Worker::copyDelta(std::vector<double>& buffer, std::vector<double>& delta){

	if(buffer.empty()) {
		buffer = delta;
	}
	else {
		for(int i = 0; i < delta.size(); i++)
			buffer[i] += delta[i];
	}
}

void Worker::applyDelta(){
	/// show delta stats
	double tt_delta_wait = 0;
	string dt = "";
	int cnt = 0;
	for (int i = deltaWaitT.size()-1; i >= 0; i--){
		if(deltaWaitT[i] > 0){
			tt_delta_wait += deltaWaitT[i];
			cnt++;
		}
		dt += std::to_string(deltaWaitT[i]) + ", ";
	}
	// VLOG_IF(iter<5 && (localID < 9 || localID % 4 == 0), 1)
	VLOG_IF(iter < 3, 1)
		<< iter << " Delta stats: " << curCalT << "||" <<  tt_delta_wait/cnt << " [" << dt << "]";

	/// apply delta to param
	DVLOG(3) << "apply buffered delta : " << bufferDelta << "\nonto: " << model.getParameter().weights;
	
	if(opt->algorighm == "km"){
		staleDelta.push_back(bufferDelta);
		if(staleDelta.size() >= staleness) {
			std::vector<double> sDelta = staleDelta.front();
			DVLOG(2) << "apply stale delta : " << sDelta << "\nonto: " << model.getParameter().weights;			
			model.accumulateParameter(sDelta, factorDelta);
			staleDelta.pop_front();
		}
	}
	else
		model.accumulateParameter(bufferDelta, factorDelta);
	
	//// reset buffer
	bfDeltaCnt = 0;
	if(isbfDeltaExt){
		isbfDeltaExt = false;
	} else {
		resetDcBuffer();
	}
}

void Worker::transmitDelta(int src, int diter){

	lock_guard<mutex> lk(mDelta);
	if(localID != 0  /// final group leader 
		&& (curHlvl == mylvl // reach transimit lvl
			|| bfDeltaCnt == int(pow(2, mylvl)) // required delta has already been received
			|| localID + bfDeltaCnt >= nWorker)) { // received enough delta 

		bufferDelta.push_back(diter);
		net->send(wm.lid2nid(dstgrpID), MType::DDelta, bufferDelta);
		if (localID * 2 == nWorker)
			deltaWaitT.push_back(tmrGlb.elapseSd());

		/// reset the buffer immidiately for later delta
		bufferDelta.clear();
		deltaIndx0.assign(nWorker, false);
		bfDeltaCnt = 0;
		++stat.n_dlt_send;
	}
}

void Worker::resetDcBuffer(){
	/// resetDcBuffer
	lock_guard<mutex> lk(mDelta);
	DVLOG_IF(bfDeltaCntExt > 0, 1) << "reset buffered delta for " << bfDeltaCntExt << " from: " << deltaIndx1
		<< "\nto: " << deltaIndx0;
	deltaWaitT.clear();

	if (bfDeltaCntExt != 0) {
		bufferDelta = move(bufferDeltaExt);
		bufferDeltaExt.clear();
		for(int i = 0; i < deltaIndx1.size(); i++){
			if(deltaIndx1[i]){
				rph.input(typeDDeltaAll, i); // add accumulated syncUnit counter
			}
		}
		deltaIndx0 = move(deltaIndx1);
		deltaIndx1.assign(nWorker, false);
	} 
	else {
		bufferDelta.clear();
		deltaIndx0.assign(nWorker, false);
		isbfDeltaExt = false;
	}
	bfDeltaCnt = bfDeltaCntExt;
	bfDeltaCntExt = 0;
}

void Worker::applyDeltaPipe(){

	DVLOG(3) << "apply buffered delta : " << bfBlkDelta[curDeltaIndex]
		<< "\nonto: " << model.getParameter().weights;
	model.accumulateParameter(bfBlkDelta[curDeltaIndex], factorDelta);

	/// reset
	// DVLOG(2) << "reset after apply delta : " << curDeltaIndex << " stale: " << stale;
	bfBlkDelta[curDeltaIndex].clear(); // reset delta buffer
	bfBlkDeltaIndex[curDeltaIndex].assign(nWorker, false); //  reset delta index buffer
	blkDeltaBFCnt[curDeltaIndex] = 0;
	stale++;
	curDeltaIndex = (curDeltaIndex+1) % blkNum;
	// DVLOG(2) << "after reset : " << curDeltaIndex << " stale: " << stale;
}

void Worker::accumulateDeltaPipe(std::vector<double>& delta, const int source, const int dIter)
{
	lock_guard<mutex> lk(mDelta);
	int indx = (dIter - stale + curDeltaIndex ) % blkNum;

	// DVLOG(2) << "accumulate delta at indx: " << indx << " dIter: " << dIter << " stale: "
	// 	<< stale << " curDeltaIndex: " << curDeltaIndex << " bfBlk size" ;
	if (!bfBlkDeltaIndex[indx][source]){ // unprocessed delta
		copyDelta(bfBlkDelta[indx], delta);
		bfBlkDeltaIndex[indx][source] = true;
	} else {
		LOG(INFO) << " WRONG number of delta received &&&&& iter: " << iter << " stale: " <<
			stale << " curDlt: " << curDeltaIndex << " from: " << source;
	}
	blkDeltaBFCnt[indx]++;
	if(blkDeltaBFCnt[indx] == nWorker) {
		Timer tmr;
		applyDeltaPipe();
		if(localID == 0)	/// send record to master only for worker 0
			sendParameter2M(); /// update parameter to master
		stat.t_par_calc += tmr.elapseSd();
	}
}

//////======  delta handler functions
void Worker::handleDelta(std::string& data, const RPCInfo & info)
{
	auto delta = deserialize<vector<double>>(data);
	int s = wm.nid2lid(info.source);
	// rph.input(typeDDeltaAll, s);
	deltaWaitT.push_back(tmrGlb.elapseSd());

	accumulateDelta(delta, s);
	//	applyDelta(delta, s);
	//	rph.input(typeDDeltaAll, s);
	//	rph.input(typeDDeltaAny, s);
	//sendReply(info);
	++stat.n_dlt_recv;
}
void Worker::handleDeltaRingcast(std::string& data, const RPCInfo & info)
{
	int src = wm.nid2lid(info.source);

	auto delta = deserialize<vector<double>>(data);
	// VLOG(2) << "w" << localID << " receive delta " << delta.size();
	int orig = delta[delta.size()-2];
	int dIter = delta[delta.size()-1];

	int nbl = (localID -1 + nWorker) % nWorker;
	int nbr = (localID +1 + nWorker) % nWorker;
	// VLOG(2) << "w" << localID << " receive delta from " << src << ", orig=" << orig << ", iter=" << diter
		// ",l=" << nbl << ",r=" << nbr
		// << ", size=" << delta.size();

	// transmit delta
	if(src == nbl && src != nbr && (dIter > iter || (dIter == iter && !deltaReceiver[orig]))) {// delta from source is not there
		// VLOG(2) << "w" << localID << " transimit delta from " << src << " to " << nbr << " org=" << orig; 
		net->send(wm.lid2nid(nbr), MType::DDelta, delta);
	}
	else if(src == nbr && src != nbl && (dIter > iter || (dIter == iter && !deltaReceiver[orig]))) {
		// VLOG(2) << "w" << localID << " transimit delta from " << src << " to " << nbl << " org=" << orig;
		net->send(wm.lid2nid(nbl), MType::DDelta, delta);
	}

	if(orig != localID && (!deltaReceiver[orig] || dIter > iter)) {
		delta.pop_back();
		delta.pop_back();
		// rph.input(typeDDeltaAll, s);
		deltaReceiver[orig] = true;
		deltaWaitT.push_back(tmrGlb.elapseSd());
		accumulateDelta(delta, orig);
		++stat.n_dlt_recv;
	}
}
void Worker::handleDeltaMltcast(std::string& data, const RPCInfo & info)
{
	int src = wm.nid2lid(info.source);

	auto delta = deserialize<vector<double>>(data);
	// VLOG(2) << "w" << localID << " receive delta " << delta.size();
	int orig = delta[delta.size()-2];
	int diter = delta[delta.size()-1];

	std::set<int> nbs;
	if (diter >= iter && !deltaReceiver[orig])
		while(nbs.size() < mltDD && nbs.size() < (nWorker-2)){
			int nb = rand() % nWorker;
			if(nb != localID && nb != src && nbs.find(nb) != nbs.end()) {
				nbs.insert(nb);
			}
		}
	// VLOG(2) << "w" << localID << " receive delta from " << src << ", orig=" << orig << ", iter=" << diter
		// ",l=" << nbl << ",r=" << nbr
		// << ", size=" << delta.size();

	// transmit delta
	for (int nb : nbs){
		// VLOG(2) << "w" << localID << " transimit delta from " << src << " to " << nbr << " org=" << orig; 
		net->send(wm.lid2nid(nb), MType::DDelta, delta);
	}

	if(orig != localID && (!deltaReceiver[orig] || diter > iter)) {
		delta.pop_back();
		delta.pop_back();
		// rph.input(typeDDeltaAll, s);
		deltaReceiver[orig] = true;
		deltaWaitT.push_back(tmrGlb.elapseSd());
		accumulateDelta(delta, orig);
		++stat.n_dlt_recv;
	}
}
void Worker::handleDeltaHrkycast(std::string& data, const RPCInfo & info)
{
	int src = wm.nid2lid(info.source);

	auto delta = deserialize<vector<double>>(data);
	// VLOG(2) << "w" << localID << " receive delta " << delta.size();

	int deltaCnt = delta[delta.size()-1];
	delta.pop_back();
	std::vector<int> origs;

	for(int i = 0; i < deltaCnt; i++){
		origs.push_back((int)delta[delta.size()-1]);
		delta.pop_back();
	}
	int diter = delta[delta.size()-1];
	delta.pop_back();
	int hlvl = delta[delta.size()-1] + 1;
	delta.pop_back();

	++stat.n_dlt_recv;
	/// accumulate delta
	accumulateDelta(delta, origs);

	// transmit buffer delta
	bufferDelta.push_back(hlvl);
	bufferDelta.push_back(diter);
	bufferDelta.insert(bufferDelta.end(), accuDelta.begin(), accuDelta.end());
	bufferDelta.push_back(accuDelta.size());
	
	int nb = localID + pow(2, hlvl);

	// reset bufferDelta
	for (int i = 0; i < 3 + accuDelta.size(); i++){
		bufferDelta.pop_back();
	}
}
void Worker::handleDeltaGrpcast(std::string& data, const RPCInfo & info)
{
	deltaWaitT.push_back(tmrGlb.elapseSd());

	int src = wm.nid2lid(info.source);
	recSrcs.push_back(src);
	auto delta = deserialize<vector<double>>(data);

	int diter = delta[delta.size()-1];
	delta.pop_back();
	// int hlvl = delta[delta.size()-1];
	VLOG(2) << " receive delta from " << src << " at iter: " << diter << " bf: " << bfDeltaCnt;
	int hlvl = id2lvl(src);
	VLOG_IF(diter != iter, 1) << "----receive accu delta from " << src << " size: " << delta.size() 
		<< " hlvl: " << hlvl << " iter: " << iter << " diter: " << diter;
	// VLOG(3) << "----receive accu delta from " << src << " size: " << delta.size() 
	// 	<< " hlvl: " << hlvl << " iter: " << iter;
	// delta.pop_back();

	++stat.n_dlt_recv;
	
	if(diter > iter){ /// avoid double reset
		isbfDeltaExt = true;
	}
	/// accumulate delta
	accumulateDelta(delta, src, hlvl, diter);
	curHlvl++;
	transmitDelta(src, diter);
}
void Worker::handleDeltaRPL(std::string& data, const RPCInfo & info)
{
	deltaWaitT.push_back(tmrGlb.elapseSd());

	int src = wm.nid2lid(info.source);
	VLOG(2) << "receive replace delta from " << src;
	if (localID % 4==0){
		for(int ss : recSrcs) {
			net->send(wm.lid2nid(ss), MType::DDeltaRPL, move(data));
		}
		if (localID + 3 < nWorker)
			net->send(wm.lid2nid(localID + 3), MType::DDeltaRPL, move(data));
	} 
	recSrcs.clear();

	auto delta = deserialize<vector<double>>(data);
	bufferDelta = move(delta);
	suDeltaAll.notify(); // notify full delta received
	// VLOG(2) << "notify suDeltaAll at " << localID;
}
void Worker::handleDeltaRPLone(std::string& data, const RPCInfo & info)
{
	deltaWaitT.push_back(tmrGlb.elapseSd());

	int src = wm.nid2lid(info.source);
	VLOG(2) << "receive replace delta from " << src;
	auto delta = deserialize<vector<double>>(data);
	bufferDelta = move(delta);
	suDeltaAll.notify(); // notify full delta received
	// VLOG(2) << "notify suDeltaAll at " << localID;
}
void Worker::handleDeltaRPLtrans(std::string& data, const RPCInfo & info)
{
	deltaWaitT.push_back(tmrGlb.elapseSd());

	int src = wm.nid2lid(info.source);
	if(src + 1 < nWorker){
		// string t=data;
		net->send2(wm.lid2nid(src + 1), MType::DDeltaRPL, move(data));
	}
	VLOG(2) << "receive replace delta from " << src;
	auto delta = deserialize<vector<double>>(data);
	bufferDelta = move(delta);
	suDeltaAll.notify(); // notify full delta received
	// VLOG(2) << "notify suDeltaAll at " << localID;
}
void Worker::handleDelta2c(std::string& data, const RPCInfo & info)
{
	auto delta = deserialize<vector<double>>(data);
	int src = wm.nid2lid(info.source);
	deltaWaitT.push_back(tmrGlb.elapseSd());

	VLOG(2) << "handle delta 2c from " << src;
	accumulateDelta(delta, src);
	if (bfDeltaCnt * 2 >= nWorker - localID % 2){
		if((src + localID) == 1){
			for(int ss = 2 ; ss < nWorker; ss += 2) {
				net->send(wm.lid2nid(ss + localID), MType::DDeltaRPL, bufferDelta);
			}
			suDeltaAll.notify();
		} else {
			VLOG(2) << "transmit the accu delta to " << 1-localID;
			net->send(wm.lid2nid(1 - localID), MType::DDelta, bufferDelta);
		}
	}

	++stat.n_dlt_recv;
}
void Worker::handleDeltaPipe(std::string& data, const RPCInfo & info)
{
	int src = wm.nid2lid(info.source);
	auto delta = deserialize<vector<double>>(data);
	int orig = delta[delta.size()-2];
	int dIter = delta[delta.size()-1];
	delta.pop_back();
	delta.pop_back();
	// deltaWaitT.push_back(tmrGlb.elapseSd());

	accumulateDeltaPipe(delta, orig, dIter);
	++stat.n_dlt_recv;
}
void Worker::handleDeltaRequest(std::string& data, const RPCInfo & info)
{
	pauseTrain();
	reqDelta = true;
}
//\\\\\\ Delta Process function end \\\\\\\\//



///////---- calculateion function start ----/////////
void Worker::updatePointer(const size_t used)
{
	DVLOG(3) << "update pointer from " << dataPointer << " by " << used;
	// dataPointer += used;
	// if(dataPointer >= trainer->pd->size())
	// 	dataPointer = 0;
	dataPointer = (dataPointer + used) % trainer->pd->size(); // round data
	stat.n_point += used;
}
void Worker::updatePointerPipe(const size_t used, const size_t blk)
{
	int ubY = 0;
	int olddp = blkPointer[blk];
	for(int yi = 0; yi <= blk; yi++) {
		ubY += blkSize[yi];
	}
	int lbY = ubY - blkSize[blk];
	int xiused = used / blkSize[blk];

	blkPointer[blk] += xiused * nny + used % blkSize[blk];
	int xi = blkPointer[blk] % nny;
	DVLOG(3) << "update pointer pipe ubY: " << ubY << " lbY " << lbY << " blkSize[blk] " 
		<< blkSize[blk]<< " xiused " << xiused << " nny " << nny << " xi " << xi;
	if (xi > ubY || xi < lbY){
		blkPointer[blk] += nny - blkSize[blk];
	}

	blkPointer[blk] %= trainer->pd->size(); // round data
	stat.n_point += used;
	DVLOG(3) << "update pointer for blk: " << blk << " from " << olddp << " to " 
		<< blkPointer[blk] << " by " << used;
}
//\\\\\\ calculateion function end \\\\\\\\//



///////---- Parameter Process function start ----/////////
void Worker::bufferParameter(Parameter & p){
	lock_guard<mutex> lk(mParam);
	bfParam = move(p);
	hasNewParam = true;
}
void Worker::applyBufferParameter(){
	//DLOG(INFO)<<"has new parameter: "<<hasNewParam;
	if(!hasNewParam)
		return;
	//DLOG(INFO)<<"before lock";
	//lock(mParam, mModel);
	lock_guard<mutex> lk(mParam);
	//DLOG(INFO)<<"after lock";
	model.setParameter(bfParam);
	DVLOG(3) << "new parameter: " << model.getParameter().size() << ", " 
			<< model.getParameter().weights;
	resumeTrain();
	//mModel.unlock();
	hasNewParam = false;
	//mParam.unlock();
}
void Worker::waitParameter(){
	suParam.wait();
	suParam.reset();
}
void Worker::fetchParmeter(){
	net->send(masterNID, MType::DRParameter, localID);
	suParam.wait();
	++stat.n_dlt_recv;
}
void Worker::sendParameter2M(){
	DVLOG(3) << "send parameter to master with: " << model.getParameter().weights;
	net->send(masterNID, MType::DParameter, model.getParameter().weights);
	++stat.n_par_send;
}
//\\\\\\ Parameter Process function end \\\\\\\\//



///////---- Centralized model start ----/////////
void Worker::syncInit()
{
	regDSPProcess(MType::DParameter, localCBBinder(&Worker::handleParameter));
}
void Worker::syncProcess()
{	
	while(!exitTrain){
		VLOG_EVERY_N(ln, 1) << "Iteration " << iter << ": calculate delta";
		Timer tmr;
		size_t cnt = localBatchSize;
		if (opt->algorighm == "lda" && iter == 0) {
			// VLOG(1) << "Inital bsize: " << localBatchSize << ", " << trainer->pd->xlength()
			// 		<< ", " << trainer->pd->size();
			cnt = trainer->pd->size();
		}
		double dly = dlyFunc();

 		tie(cnt, bfDelta) = trainer->batchDelta(allowTrain, dataPointer, cnt, dly, -1);
		updatePointer(cnt);
		// double calT = tmr.elapseSd();
		// if (calT < forceCalT/nWorker) { /// force homo case
		// 	sleep(forceCalT/nWorker - calT);
		// }
		curCalT = tmr.elapseSd();
		///// add delay
		// if (localID < delayWorkers){
		// 	VLOG_IF(iter<2, 1) << "sleep for worker " << localID << ": " << interval; 
		// 	sleep(interval * curCalT);
		// 	curCalT += curCalT * interval;
		// }
		VLOG_IF(iter<5 && (localID < 3), 1) << "iter " << iter << " CALT: " << curCalT 
			<< "; unit dp " << cnt << " : " << curCalT/cnt  << " dly: " << dly;
		
		stat.t_dlt_calc += tmr.elapseSd();
		// VLOG_EVERY_N(ln, 2) << "  send delta";
		tmr.restart();
		sendDelta(bfDelta);
		double sendT = tmr.elapseSd();
		tmr.restart();

		if(exitTrain==true){
			break;
		}
		// VLOG_EVERY_N(ln, 2) << "  wait for new parameter";
		waitParameter();

		if(exitTrain==true){
			break;
		}
		double waitT = tmr.elapseSd();
		///// add waitting delay
		// if (waitT <  calT * delayWorkers / 100){
		// 	VLOG_IF(iter<2, 1) << "sleep for worker " << localID << ": " << interval; 
		// 	sleep(delayWorkers * waitT);
		// }
		stat.t_par_wait += tmr.elapseSd();
		tmr.restart();
		applyBufferParameter();
		stat.t_par_calc += tmr.elapseSd();

		// VLOG_IF(iter<50 && (localID < 9), 1) << "CALT: " << curCalT 
		// 		<< "; unit dp " << cnt << " : " << curCalT/cnt << " WT: " << waitT << " SDT: " << sendT;
		++iter;
	}
}
void Worker::asyncInit()
{
	regDSPProcess(MType::DParameter, localCBBinder(&Worker::handleParameterAsync));
}
void Worker::asyncProcess()
{
	while(!exitTrain){
		VLOG_EVERY_N(ln, 1) << "Iteration " << iter << ": calculate delta";
		Timer tmr;
		// DVLOG(3) << "current parameter: " << model.getParameter().weights;
		// try to use localBatchSize data-points, the actual usage is returned via cnt 
		size_t cnt = localBatchSize;
		if (opt->algorighm == "lda" && iter == 0) {
			// VLOG(1) << "Inital bsize: " << localBatchSize << ", " << trainer->pd->xlength()
			// 		<< ", " << trainer->pd->size();
			cnt = trainer->pd->size();
		}
		double dly = dlyFunc();

		tie(cnt, bfDelta) = trainer->batchDelta(allowTrain, dataPointer, cnt, dly, -1);
		updatePointer(cnt);

		// double calT = tmr.elapseSd();
		// if (calT < forceCalT) {
		// 	sleep(forceCalT - calT);
		// }
		curCalT = tmr.elapseSd();
		///// add delay
		// if (localID < delayWorkers){
		// 	VLOG_IF(iter<2, 1) << "sleep for worker " << localID << ": " << interval; 
		// 	sleep(interval * curCalT);
		// 	curCalT += curCalT * interval;
		// }
		VLOG_IF(iter<10 && (localID < 4), 1) << "CALT: " << curCalT 
				<< "; unit dp " << cnt << " : " << curCalT/cnt;
		stat.t_dlt_calc += tmr.elapseSd();
		
		VLOG_EVERY_N(ln, 2) << "  send delta";
		tmr.restart();
		sendDelta(bfDelta, paramVersion);
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
void Worker::progAsyncInit()
{
	regDSPProcess(MType::DParameter, localCBBinder(&Worker::handleParameterProg));
	regDSPProcess(MType::DDeltaReq, localCBBinder(&Worker::handleDeltaRequest));
}
void Worker::progAsyncProcess()
{	
	double sendT = 0;
	double dly = dlyFunc();

	Timer TTtmr;
	double t_ws = 0, t_ws_prev = TTtmr.elapseSd();
	// size_t remaincnt = localBatchSize;
	while(!exitTrain){
		Timer tmr;
		// DVLOG(3) << "current parameter: " << model.getParameter().weights;
		// try to use localBatchSize data-points, the actual usage is returned via cnt 
		size_t cnt = 0;
		
		int remainCnt = reportSize;
		if (opt->mode.find("pasp1") !=std::string::npos)
			remainCnt = reportSize - curCnt;

		tie(cnt, bfDelta) = trainer->batchDelta(allowTrain, dataPointer, remainCnt, dly, -1);
		t_ws_prev = TTtmr.elapseSd();
		updatePointer(cnt);

		copyDelta(bufferDeltaExt, bfDelta); 
		curCnt += cnt;
		curCalT += tmr.elapseSd();
		stat.t_dlt_calc += tmr.elapseSd();
		
		if (!allowTrain){
			VLOG_EVERY_N(ln, 1) << "Iteration " << iter << ": calculate delta";
			tmr.restart();
			if(reqDelta){
				DVLOG(3) << "send delta pasp: " << curCnt << "; " << bfDelta.size() << "; " << bfDelta;
				sendDelta(bufferDeltaExt, curCnt);
				reqDelta = false;
				++iter;
				VLOG_IF(iter<5 && (localID < 3), 1) << "iter " << iter << " CALT: " << curCalT 
					<< "; unit dp " << curCnt << " : " << curCalT/curCnt  << " dly: " << dly
					<< " sendT: " << sendT/curCnt;
				bufferDeltaExt.clear();
				curCnt = 0;
				curCalT = 0;

				/// worker send report time;
				t_wdelta = TTtmr.elapseSd() - t_ws_prev;
				t_ws_prev = TTtmr.elapseSd();
			}
			if(hasNewParam){
				tmr.restart();
				applyBufferParameter();
				hasNewParam = false;
				double updateParamT = tmr.elapseSd();
				stat.t_par_calc += tmr.elapseSd();
				VLOG_IF(iter<5 && (localID < 3), 1) << "iter " << iter << " interrupt CALT: " << curCalT 
						<< "; unit dp " << cnt << " : " << curCalT/cnt << " dly: " << dly
						<< " ParamT: " << updateParamT;
				
				double dly = dlyFunc();
			}
			allowTrain = true;

		} else{
			tmr.restart();
			// VLOG_EVERY_N(ln, 2) << "  send delta";
			if (opt->mode.find("pasp1") !=std::string::npos ||
				opt->mode.find("pasp5") !=std::string::npos){
				sendDelta(bfDelta, curCnt);
				VLOG_IF(iter<5 && (localID < 3), 1) << "iter " << iter << " CALT: " << curCalT 
					<< "; unit dp " << curCnt << " : " << curCalT/curCnt  << " dly: " << dly
					<< " sendT: " << sendT/curCnt;
				bfDelta.clear();
				curCnt = 0;
			}
			else {// if (opt->mode.find("pasp") !=std::string::npos)
				vector<double> report;
				report.push_back(cnt);
				report.push_back(bufferDeltaExt.back());
				report.push_back(curCalT/curCnt);
				report.push_back(t_ws);
				report.push_back(t_wdelta);
				sendReport(report);
			}
			if (opt->mode.find("pasp2") !=std::string::npos)
				model.accumulateParameter(bfDelta, factorDelta);

			sendT += tmr.elapseSd();
			stat.t_dlt_send += tmr.elapseSd();
			
			/// worker send report time;
			t_ws = TTtmr.elapseSd() - t_ws_prev;
			t_ws_prev = TTtmr.elapseSd();
			// remaincnt = localBatchSize;
		}
		if(exitTrain==true){
			break;
		}
	}
}
/*** void Worker::paspProcess()
{
	size_t remaincnt = localBatchSize;
	while(!exitTrain){
		VLOG_EVERY_N(ln, 1) << "Iteration " << iter << ": calculate delta";
		Timer tmr;
		// DVLOG(3) << "current parameter: " << model.getParameter().weights;
		// try to use localBatchSize data-points, the actual usage is returned via cnt 
		size_t cnt = 0;
		double dly = lamda > 0 ? delayArr[rand() % range] : 0; /// random seed......
		tie(cnt, bfDelta) = trainer->batchDelta(allowTrain, dataPointer, remaincnt, dly, true);
		curCnt += cnt;
		remaincnt -= cnt;
		updatePointer(cnt);

		curCalT += tmr.elapseSd();
		stat.t_dlt_calc += tmr.elapseSd();
		
		if (!allowTrain){
			tmr.restart();
			applyBufferParameter();
			allowTrain = true;
			double updateParamT = tmr.elapseSd();
			stat.t_par_calc += tmr.elapseSd();
			VLOG_IF(iter<10 && (localID < 9), 1) << "iter " << iter << " interrupt CALT: " << curCalT 
				<< "; unit dp " << cnt << " : " << curCalT/cnt << " dly: " << dly
				<< " ParamT: " << updateParamT;
		} else{
			tmr.restart();
			VLOG_EVERY_N(ln, 2) << "  send delta";
			sendDelta(bfDelta, curCnt);
			double sendT = tmr.elapseSd();
			stat.t_dlt_send += tmr.elapseSd();
			VLOG_IF(iter<10 && (localID < 9), 1) << "iter " << iter << " CALT: " << curCalT 
				<< "; unit dp " << curCnt << " : " << curCalT/curCnt  << " dly: " << dly
				<< " sendT: " << sendT;
			curCalT = 0;
			curCnt = 0;
			remaincnt = localBatchSize;
		}
		if(exitTrain==true){
			break;
		}
		++iter;
	}
}***/

void Worker::papProcess()
{	
	double sendT = 0;
	double dly = dlyFunc();

	Timer TTtmr;
	double t_ws = 0, t_ws_prev = TTtmr.elapseSd();
	while(!exitTrain){
		Timer tmr;
		// try to use localBatchSize data-points, the actual usage is returned via cnt 
		size_t cnt = 0;

		tie(cnt, bfDelta) = trainer->batchDelta(allowTrain, dataPointer, reportSize, dly, -1);
		updatePointer(cnt); // cnt for this report
		copyDelta(bufferDeltaExt, bfDelta); // accumulate delta before request

		curCnt += cnt; // accumulate count for this mini-batch
		curCalT += tmr.elapseSd();
		stat.t_dlt_calc += tmr.elapseSd();
		t_ws_prev = TTtmr.elapseSd();
		
		if (!allowTrain){
			VLOG_EVERY_N(ln, 1) << "Iteration " << iter << ": calculate delta";
			tmr.restart();
			if(reqDelta){
				DVLOG(3) << "send delta pasp: " << curCnt << "; " << bfDelta.size() << "; " << bfDelta;
				sendDelta(bufferDeltaExt, curCnt);
				reqDelta = false;
				++iter;
				VLOG_IF(iter<5 && (localID < 3), 1) << "iter " << iter << " CALT: " << curCalT 
					<< "; unit dp " << curCnt << " : " << curCalT/curCnt  << " dly: " << dly
					<< " sendT: " << sendT/curCnt;
				bufferDeltaExt.clear();
				curCnt = 0;
				curCalT = 0;

				/// worker send report time;
				t_wdelta = TTtmr.elapseSd() - t_ws_prev;
				t_ws_prev = TTtmr.elapseSd();
			}
			if(hasNewParam){
				tmr.restart();
				applyBufferParameter();
				hasNewParam = false;
				double updateParamT = tmr.elapseSd();
				stat.t_par_calc += tmr.elapseSd();
				VLOG_IF(iter<5 && (localID < 3), 1) << "iter " << iter << " interrupt CALT: " << curCalT 
						<< "; unit dp " << cnt << " : " << curCalT/cnt << " dly: " << dly
						<< " ParamT: " << updateParamT;
				
				double dly = dlyFunc();
			}
			allowTrain = true;

		} else{
			tmr.restart();
			// VLOG_EVERY_N(ln, 2) << "  send delta";
			if (opt->mode.find("pasp1") !=std::string::npos ||
				opt->mode.find("pasp5") !=std::string::npos){
				sendDelta(bfDelta, curCnt);
				VLOG_IF(iter<5 && (localID < 3), 1) << "iter " << iter << " CALT: " << curCalT 
					<< "; unit dp " << curCnt << " : " << curCalT/curCnt  << " dly: " << dly
					<< " sendT: " << sendT/curCnt;
				bfDelta.clear();
				curCnt = 0;
			}
			else {// if (opt->mode.find("pasp") !=std::string::npos)
				vector<double> report;
				report.push_back(cnt);
				report.push_back(bufferDeltaExt.back());
				report.push_back(curCalT/curCnt);
				report.push_back(t_ws);
				report.push_back(t_wdelta);
				sendReport(report);
			}
			if (opt->mode.find("pasp2") !=std::string::npos)
				model.accumulateParameter(bfDelta, factorDelta);

			sendT += tmr.elapseSd();
			stat.t_dlt_send += tmr.elapseSd();
			
			/// worker send report time;
			t_ws = TTtmr.elapseSd() - t_ws_prev;
			t_ws_prev = TTtmr.elapseSd();
			// remaincnt = localBatchSize;
		}
		if(exitTrain==true){
			break;
		}
	}
}

void Worker::fsbInit()
{
	// regDSPProcess(MType::DParameter, localCBBinder(&Worker::handleParameter));
	regDSPProcess(MType::DParameter, localCBBinder(&Worker::handleParameterFsb));
}
void Worker::fsbProcess()
{
	while(!exitTrain){
		// if(allowTrain == false){
		// 	sleep();
		// 	continue;
		// }
		VLOG_EVERY_N(ln, 1) << "Iteration " << iter << ": calculate delta";
		Timer tmr;
		size_t cnt;
		// try to use localBatchSize data-points, the actual usage is returned via cnt 
		tie(cnt, bfDelta) = trainer->batchDelta(allowTrain, dataPointer, localBatchSize, true);
		updatePointer(cnt);
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
		if(exitTrain==true){
			break;
		}
		stat.t_par_wait += tmr.elapseSd();
		tmr.restart();
		applyBufferParameter();
		// resumeTrain();
		stat.t_par_calc += tmr.elapseSd();
		++iter;
	}

	// localBatchSize = trainer->pd->size();
	// const size_t n = model.paramWidth();
	// while(!exitTrain){
	// 	//if(allowTrain == false){
	// 	//	sleep();
	// 	//	continue;
	// 	//}
	// 	VLOG_EVERY_N(ln, 1) << "Iteration " << iter << ": calculate delta";
	// 	Timer tmr;
	// 	size_t cnt = 0;
	// 	bfDelta.assign(n, 0.0);
	// 	while (exitTrain == false && allowTrain) {
	// 		vector<double> tmp;
	// 		size_t c;
	// 		// try to use localBatchSize data-points, the actual usage is returned via cnt
	// 		tie(c, tmp) = trainer->batchDelta(allowTrain, dataPointer, localBatchSize, true);
	// 		accumulateDelta(tmp);
	// 		updatePointer(c);
	// 		cnt += c;
	// 	}
	// 	stat.t_dlt_calc += tmr.elapseSd();
	// 	VLOG_EVERY_N(ln, 2) << "  calculate delta with " << cnt << " data points";
	// 	VLOG_EVERY_N(ln, 2) << "  send delta";
	// 	tmr.restart();
	// 	sendDelta(bfDelta);
	// 	if(exitTrain==true){
	// 		break;
	// 	}
	// 	VLOG_EVERY_N(ln, 2) << "  wait for new parameter";
	// 	waitParameter();
	// 	resumeTrain();
	// 	if(exitTrain==true){
	// 		break;
	// 	}
	// 	stat.t_par_wait += tmr.elapseSd();
	// 	tmr.restart();
	// 	applyBufferParameter();
	// 	resumeTrain();
	// 	stat.t_par_calc += tmr.elapseSd();
	// 	++iter;
	// }
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
			tie(cnt, tmp) = trainer->batchDelta(allowTrain, dataPointer, left, false);
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
//\\\\\\ Centralized model end \\\\\\//



///////---- old DeCentralized model start ----/////////
void Worker::dcSyncProcess()
{	
	while(!exitTrain && iter <= opt->tcIter){
		if(allowTrain.load() == false){
			sleep();
			continue;
		}
		VLOG_EVERY_N(ln, 1) << "Iteration " << iter << ": calculate delta";
		Timer tmr;
		bfDelta = trainer->batchDelta(dataPointer, localBatchSize, true);
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
//\\\\\\ old DeCentralized model end \\\\\\//


///////---- Basic Handler function start ----/////////
Worker::callback_t Worker::localCBBinder(
	void (Worker::*fp)(std::string&, const RPCInfo&))
{
	return bind(fp, this, placeholders::_1, placeholders::_2);
}
void Worker::registerHandlers(){
	regDSPProcess(MType::CReply, localCBBinder(&Worker::handleReply));
	regDSPProcess(MType::CWorkers, localCBBinder(&Worker::handleWorkerList));
	regDSPProcess(MType::CTrainPause, localCBBinder(&Worker::handlePause));
	regDSPProcess(MType::CTrainContinue, localCBBinder(&Worker::handleContinue));
	regDSPImmediate(MType::CTerminate, localCBBinder(&Worker::handleTerminate));

	regDSPImmediate(MType::CTrainInterval, localCBBinder(&Worker::handleInterval));
	//regDSPProcess(MType::DParameter, localCBBinder(&Worker::handleParameter));
	//regDSPProcess(MType::DDelta, localCBBinder(&Worker::handleDelta));

	//addRPHAnySU(MType::CWorkers, suOnline);
	//addRPHAnySU(MType::DParameter, suParam);
	addRPHAnySU(MType::CXLength, suXlength);
}

void Worker::handleReply(std::string& data, const RPCInfo& info) {
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
void Worker::handleWorkerList(std::string& data, const RPCInfo & info)
{
	DLOG_IF(localID < 4, INFO) << "receive worker list";
	Timer tmr;
	auto res = deserialize<vector<pair<int, int>>>(data);
	stat.t_data_deserial += tmr.elapseSd();
	for(auto& p : res){
		DLOG_IF(nWorker<6, INFO)<<"register nid "<<p.first<<" with lid "<<p.second;
		wm.registerID(p.first, p.second);
	}
	//rph.input(MType::CWorkers, info.source);
	suOnline.notify();
	sendReply(info);
}
void Worker::handleParameter(std::string& data, const RPCInfo & info)
{
	VLOG(3) << "Receive parameter from " << info.source;
	Timer tmr;
	auto weights = deserialize<vector<double>>(data);

	stat.t_data_deserial += tmr.elapseSd();
	Parameter p;
	p.set(move(weights));
	// VLOG(3) << "set parameter to bfParam ";
	bufferParameter(p);
	VLOG(3) << "notify sync unit";
	suParam.notify();
	//sendReply(info);
	++stat.n_par_recv;
}
void Worker::handleParameterAsync(std::string& data, const RPCInfo & info)
{
	Timer tmr;
	auto weights = deserialize<vector<double>>(data);
	paramVersion = weights.back(); /// update current param version
	weights.pop_back();
	VLOG(3) << "Receive parameter from " << info.source << ", w: " << paramVersion;

	stat.t_data_deserial += tmr.elapseSd();
	Parameter p;
	p.set(move(weights));
	// VLOG(3) << "set parameter to bfParam ";
	bufferParameter(p);
	VLOG(3) << "notify sync unit";
	suParam.notify();
	//sendReply(info);
	++stat.n_par_recv;
}

void Worker::handleParameterProg(std::string& data, const RPCInfo & info)
{
	Timer tmr;
	auto weights = deserialize<vector<double>>(data);
	stat.t_data_deserial += tmr.elapseSd();
	/// adjust report size
	if (opt->mode.find("pasp6") !=std::string::npos){
		VLOG_IF(iter < 4, 1) << "glbBS: " << weights.back();
		reportSize = weights.back()/4; /// update current param version
		weights.pop_back();
	}

	Parameter p;
	p.set(move(weights));
	bufferParameter(p);
	suParam.notify();
	//sendReply(info);
	// break the trainning and apply the received parameter (in main thread)
	hasNewParam = true;
	pauseTrain();
	//applyBufferParameter();
	++stat.n_par_recv;
}
void Worker::handleParameterFab(std::string& data, const RPCInfo & info)
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
void Worker::handleParameterFsb(std::string& data, const RPCInfo & info)
{
	auto weights = deserialize<vector<double>>(data);
	Parameter p;
	p.set(move(weights));
	bufferParameter(p);
	suParam.notify();
	//sendReply(info);
	// continue training
	// resumeTrain();
	++stat.n_par_recv;
}
void Worker::handlePause(std::string& data, const RPCInfo & info)
{
	pauseTrain();
	sendReply(info);
}
void Worker::handleContinue(std::string& data, const RPCInfo & info)
{
	resumeTrain();
	sendReply(info);
}
void Worker::handleTerminate(std::string& data, const RPCInfo & info)
{
	exitTrain = true;
	pauseTrain(); // in case if the system is calculating delta
	suParam.notify(); // in case if the system just calculated a delta (is waiting for new parameter)
	suDeltaAll.notify(); // in case the worker is waiting other parameters
	sendReply(info);
}
void Worker::handleInterval(std::string& data, const RPCInfo& info){
	double nInterval = deserialize<double>(data);
	interval = interval > 99999 ? nInterval : (interval + nInterval)/2;
	localBatchSize *= 2;
	VLOG(1) << "NNNNew interval " << interval;
}
//\\\\\\ Basic Handler function end \\\\\\\\//


///// util functions /////
size_t Worker::id2lvl(const size_t id){
	size_t i = 0;
	for ( ;(id % int(pow(2, i+1))) == 0; i++){}
	return i;
}

int Worker::dstGrpID(const size_t id, const size_t lvl){
	return id - id % int(pow(2, lvl+1));
}

double Worker::dlyFunc(){
	double dly = lamda > 0 ? distribution(gen) : 0; /// random seed......
	if (lamda > 100){
		if (localID <= (lamda - 100) * nWorker / 100)
			dly = range;
	}
	else
		dly = dly > 1 || dly < 0.1 ? 0 : dly * range;
	return dly;
}