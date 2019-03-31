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

	ln = opt->logIter;
	logName = "W"+to_string(localID);
	setLogThreadName(logName);

	if (opt->algorighm == "km") {
		trainer = new EM;
	}else if (opt->algorighm.find("nmf") !=std::string::npos) {
		trainer = new EM;
		trainer->setRate(opt->lrate);
	}else {
		trainer = new GD;
		trainer->setRate(opt->lrate);
	}

	/// for dc cache
	deltaIndx0.assign(nWorker, false);
	deltaIndx1.assign(nWorker, false);
	deltaReceiver.assign(nWorker, false);
	tmrGlb.restart();
	curCalT = 0;
	deltaWaitT = std::vector<double>();
	curHlvl = 0;
	mylvl = localID == 0? nWorker :id2lvl(localID);
	dstgrpID = dstGrpID(localID, mylvl);

	/// multicast
	mltDD = 3; /// multicast degree

	if(opt->mode == "sync"){
		syncInit();
	} else if(opt->mode == "async"){
		asyncInit();
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
void Worker::run()
{
	LOG(INFO) << "register handlers";
	registerHandlers();  // register message with function, function need to wrap up
	startMsgLoop(logName+"-MSG"); // make a new thread to record messages

	LOG(INFO) << "start";
	DLOG_IF(localID < 4, INFO) << "send online message";
	sendOnline();

	DLOG_IF(localID < 4, INFO) << "waiting worker list";
	waitWorkerList();
	DLOG_IF(localID < 4, INFO) << "send x length " << trainer->pd->xlength();
	sendXLength(); // x dimension
	DLOG_IF(localID < 4, INFO) << "waiting init parameter";
	waitParameter();
	// DLOG(INFO) << "got init parameter";
	model.init(opt->algorighm, trainer->pd->xlength(), opt->algParam);
	trainer->bindModel(&model); /// move
	trainer->initState(1);

	applyBufferParameter();
	resumeTrain();

	DLOG_IF(localID < 4, INFO) << "start training with mode: " << opt->mode << ", local batch size: " << localBatchSize;
	iter = 0;
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
	// } else if(opt->mode == "dcsync"){
	// 	dcSyncProcess();
	} else if(opt->mode.find("dc") !=std::string::npos){
		LOG(INFO) << "dcFsb Process ";
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
		double updateDpT = tmr.elapseSd() - curCalT;
		updatePointer(cnt);
		VLOG_IF(iter<2, 1) << "unit dp cal time for " << cnt << " : " << curCalT/cnt;
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
		else if(opt->mode.find("grp") !=std::string::npos)
			grpcastDelta(lclDelta);
		stat.t_dlt_send += tmr.elapseSd();

		double offset = tmrGlb.elapseSd(); // recompute the early received delta
		for(double tt : deltaWaitT){
			tt -= offset;
		}
		
		tmr.restart();
		tmrGlb.restart(); // for monitoring the delta ariving time
		accumulateDelta(lclDelta, (int)localID);
		stat.t_dlt_accumLcl += tmr.elapseSd();

		if(exitTrain){ break; }
		// VLOG_EVERY_N(ln, 2) << "  DC: wait for delta from all other workers";
		tmr.restart();
		waitDeltaFromAll();
		double wT = tmr.elapseSd();
		stat.t_par_wait += tmr.elapseSd();
		if(exitTrain){ break; }

		tmr.restart();
		// VLOG(2) << "GO applyDelta ";
		applyDelta();
		resumeTrain();
		// VLOG(2) << "Resume train ";
		if(localID == 0)	/// send record to master only for worker 0
			sendParameter2M(); /// update parameter to master
		stat.t_par_calc += tmr.elapseSd();

		// VLOG(2) << " calculate time: " << curCalT << " for dp: " << cnt 
		// 	<< " update pointer time: " << updateDpT
		// 	<< " wait delta time: " << wT << " update param time: " << tmr.elapseSd();

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
	localBatchSize = opt->batchSize / nWorker;
	if(opt->batchSize % nWorker > localID)
		++localBatchSize;
	if(localBatchSize <= 0)
		localBatchSize = 1;
}
void Worker::initPipeBlk()
{
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
	if(opt->algorighm == "km") {
		/// send k dp as candidate of global centroids
		std::vector<double> kCentroids;
		int k = stoi(opt->algParam);
		int i = 0;
		for(; i < k && i < trainer->pd->size(); i++){
			std::vector<double> OneDp = trainer->pd->get(i).x;
			kCentroids.insert(kCentroids.end(), OneDp.begin(), OneDp.end());
			kCentroids.push_back(1); // for cluster counts
		}
		kCentroids.push_back(i);
		net->send(masterNID, MType::CXLength, kCentroids);
	}
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
void Worker::grpcastDelta(std::vector<double>& delta)
{	
	// if(localID % 2 != 0) { // none leader node

	// 	VLOG(3) << "send delta from " << localID << " to grpLeader " << localID-1;
	// 	delta.push_back(0); // add hierarchy level
	// 	// delta.push_back(iter); // add delta iter #
	// 	net->send(wm.lid2nid(localID-1), MType::DDelta, delta);
	// 	++stat.n_dlt_send;
	// 	VLOG(3) << "delta sent to " << localID-1;
	// }
	if(localID!=0 && (mylvl == 0 || localID + 1 >= nWorker)){ // # of workers is odd
		delta.push_back(mylvl); // add hierarchy level
		delta.push_back(iter); // add iter #
		net->send(wm.lid2nid(dstgrpID), MType::DDelta, delta);
		++stat.n_dlt_send;
		VLOG(2) << "delta sent to " << dstgrpID << " w hlvl: " << mylvl;
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
void Worker::accumulateDelta(std::vector<double>& delta, const int source, const size_t hlvl)
{
	lock_guard<mutex> lk(mDelta);
	int powhlvl = pow(2, hlvl);
	// VLOG(2) << "accu delta from " << source << " with pow hlvl " << powhlvl << " indx size " << deltaIndx0.size();
	if (deltaIndx0[source]) { // if a delta from source is already there
		copyDelta(bufferDeltaExt, delta);
		for (int i = 0; i < powhlvl && source+i < nWorker; i++) {
			deltaIndx1[source + i] = true;
		}
	}
	else {
		DVLOG_IF(deltaIndx1[source], 1) << " Dam WWWTTTFFFF number of delta applied &&&&&&&";
		copyDelta(bufferDelta, delta);
		int i = 0;
		int newcnt = powhlvl + source > nWorker ? powhlvl : nWorker - source;
		bfDeltaCnt += newcnt;
		VLOG(2) << "====accu delta from " << source << " with pow hlvl " << powhlvl 
			<< " delta size " << newcnt << " bfDeltaCnt " << bfDeltaCnt;
			
		if(bfDeltaCnt == nWorker) {
			// VLOG(2) << "broadcast rpl delta from " << localID;
			// for(int i = 1; i < nWorker; i++) {
			// 	net->send(wm.lid2nid(i), MType::DDeltaRPL, bufferDelta);
			// }
			net->broadcast(MType::DDeltaRPL, bufferDelta);
		}
		for ( ; i < powhlvl && source+i < nWorker; i++) {
			deltaIndx0[source + i] = true;
			rph.input(typeDDeltaAll, source+i); // trigger the syncUnit counter
		}
	}
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
		buffer = move(delta);
	}
	else {
		for(int i = 0; i < delta.size(); i++)
			buffer[i] += delta[i];
	}
}
void Worker::applyDelta(){
	// #if DEBUG
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
	VLOG_IF(iter<5, 1) << "Delta stats: " << curCalT << "||" <<  tt_delta_wait/cnt << " [" << dt;
	// #endif

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

	deltaWaitT.clear();
	bfDeltaCnt = 0;
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
void Worker::handleDelta(const std::string & data, const RPCInfo & info)
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
void Worker::handleDeltaRingcast(const std::string & data, const RPCInfo & info)
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
void Worker::handleDeltaMltcast(const std::string & data, const RPCInfo & info)
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
void Worker::handleDeltaHrkycast(const std::string & data, const RPCInfo & info)
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
void Worker::handleDeltaGrpcast(const std::string & data, const RPCInfo & info)
{
	deltaWaitT.push_back(tmrGlb.elapseSd());

	int src = wm.nid2lid(info.source);
	auto delta = deserialize<vector<double>>(data);
	// VLOG(2) << "w" << localID << " receive delta " << delta.size();

	int diter = delta[delta.size()-1];
	delta.pop_back();
	int hlvl = delta[delta.size()-1];
	VLOG_IF(diter != iter, 1) << "----receive accu delta from " << src << " size: " << delta.size() 
		<< " hlvl: " << hlvl << " iter: " << iter;
	VLOG(3) << "----receive accu delta from " << src << " size: " << delta.size() 
		<< " hlvl: " << hlvl << " iter: " << iter;
	delta.pop_back();

	++stat.n_dlt_recv;

	/// accumulate delta
	accumulateDelta(delta, src, hlvl);
	curHlvl++;

	// transmit buffer delta
	if(localID!=0 && (curHlvl == mylvl || localID + int(pow(2, curHlvl)) >= nWorker)) {
		// VLOG(2) << "transmit delta from " << localID << " to: " << dstgrpID << " hlvl: " << hlvl;
		bufferDelta.push_back(mylvl);
		bufferDelta.push_back(diter);
		net->send(wm.lid2nid(dstgrpID), MType::DDelta, bufferDelta);
		++stat.n_dlt_send;
	}
}
void Worker::handleDeltaRPL(const std::string & data, const RPCInfo & info)
{
	deltaWaitT.push_back(tmrGlb.elapseSd());
	int src = wm.nid2lid(info.source);
	// VLOG(2) << "receive replace delta from " << src;
	auto delta = deserialize<vector<double>>(data);
	bufferDelta = move(delta);
	suDeltaAll.notify(); // notify full delta received
	// VLOG(2) << "notify suDeltaAll at " << localID;
}
void Worker::handleDeltaPipe(const std::string & data, const RPCInfo & info)
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
	DVLOG(3) << "apply parameter: " << bfParam.weights;
	model.setParameter(bfParam);
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
		bfDelta = trainer->batchDelta(dataPointer, localBatchSize, true);
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
		bfDelta = trainer->batchDelta(dataPointer, localBatchSize, true);
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
	void (Worker::*fp)(const std::string&, const RPCInfo&))
{
	return bind(fp, this, placeholders::_1, placeholders::_2);
}
void Worker::registerHandlers(){
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
		DLOG_IF(nWorker<6, INFO)<<"register nid "<<p.first<<" with lid "<<p.second;
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
void Worker::handleParameterFsb(const std::string & data, const RPCInfo & info)
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