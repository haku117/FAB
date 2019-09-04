#include "Option.h"
#include <vector>
#include <iostream>
#include "util/Util.h"

using namespace std;

// should be called after set nw outside
static bool dummySet(Option& opt, int argc, char * argv[]){
	opt.mode = argc > 1 ? argv[1] : "sync";
	opt.fabWait = false;
	//opt.fnData = argc > 2 ? argv[2] : "E:/Code/FSB/dataset/affairs.csv";
	string prefix = "E:/Code/FSB/";
	opt.fnData = argc > 2 ? argv[2] : prefix + "/dataset/affairs.csv";
	opt.fnOutput = argc > 3 ? argv[3] : prefix + "/result/" + opt.mode + "-" + to_string(opt.nw) + ".csv";
	opt.idSkip = argc > 4 ? getIntList(argv[4]) : vector<int>{ 1 };
	opt.idY = argc > 5 ? getIntList(argv[5]) : vector<int>{ 9 };
	opt.doNormalize = argc > 6 ? beTrueOption(argv[6]) : true;
	opt.lrate = argc > 7 ? stod(argv[7]) : 0.1;
	opt.batchSize = argc > 8 ? stoiKMG(argv[8]) : 500;
	opt.tcIter = argc > 9 ? stoiKMG(argv[9]) : 10;
	opt.tcTime = argc > 10 ? stod(argv[10]) : 10.0;
	opt.tcDiff = 1e-5;
	opt.arvIter = argc > 11 ? stoiKMG(argv[11]) : 1000;
	opt.arvTime = argc > 12 ? stod(argv[12]) : 0.5;
	opt.logIter = argc > 13 ? stoiKMG(argv[13]) : 1000;
	opt.header = false;
	opt.algorighm = "mlp";
	opt.algParam = "3-5-1";
	return true;
}

bool Option::parse(int argc, char * argv[], const size_t nWorker)
{
	this->nw = nWorker;
	//return dummySet(*this, argc, argv);

	int idx = 1;
	int optIdx = 13;
	bool flag = false;
	// parse
	if(argc < optIdx)
		return false;
	try{
		mode = argv[idx++];
		algorighm = argv[idx++];
		algParam = argv[idx++];
		dataSize = stoi(argv[idx++]);
		straggler = argv[idx++];
		reptr = stoi(argv[idx++]);
		fabWait = false;

		fnData = argv[idx++];
		fnOutput = argv[idx++];
		idSkip = getIntList(argv[idx++]);
		idY = getIntList(argv[idx++]);

		doNormalize = beTrueOption(argv[idx++]);
		lrate = stod(argv[idx++]);
		batchSize = stoulKMG(argv[idx++]);
		// finish condition
		tcIter = stoulKMG(argv[idx++]); // maximum iteration
		tcTime = stod(argv[idx++]); // maximum training time
		//tcDiff = stod(argv[idx++]); // minimum improvement cross iterations
		if(argc > optIdx++) {
			arvIter = stoiKMG(argv[idx++]);
		}
		if(argc > optIdx++){
			arvTime = stod(argv[idx++]);
		}
		if(argc > optIdx++){
			// cout << "param: " << idx << " -> " << argv[idx] << endl;
			logIter = stoiKMG(argv[idx++]);
			// cout << "param: " << idx << " -> " << argv[idx] << "; ";
		}
	} catch(exception& e){
		cerr << "Cannot parse the " << idx << "-th parameter: " << argv[idx-1] 
			<< " prev: " << argv[idx-2] << endl;
		cerr << "Error message: " << e.what() << endl;
		return false;
	}
	// check and preprocess
	if(!preprocessMode()){
		cerr << "Error: mode not supported." << endl;
		return false;
	}
	if(!processAlgorithm()){
		cerr << "Error: algorithm not supported" << endl;
		return false;
	}
	// error handling

	return true;
}

void Option::showUsage() const {
	cout << "usage: <mode> <alg> <param> <data-file> <output-file> <id-skip> <id-y> <normalize>"
		" <lrate> <batch-size> <term-iter> <term-time> [arv-iter=1000] [arv-time=0.5] [log-iter=1000]" << endl;
	//cout << "usage: <algorithm> <mode> <data-file> <output-file> <id-skip> <id-y> <nw> <batch-size> <term-iter> <term-time>" << endl;
	cout << "  <mode>: sync, async, fsb, fab"
		<< "  <alg>: algorithm name. Support: lr, mlp, cnn.\n"
		<< "  <param>: parameter of the algorithm, usually the shape of the algorithm.\n"
		<< "  <id-skip>: a list separated with space or comma."
		<< endl;
}

bool Option::preprocessMode(){
	for(char& ch : mode){
		if(ch >= 'A' && ch <= 'Z')
			ch += 'a' - 'A';
	}
	if(mode.find("sync")!=std::string::npos || mode.find("asyc")!=std::string::npos 
		|| mode == "fsb" || mode == "fab" || mode.find("pasp")!=std::string::npos 
		|| mode == "dcsync" || mode.find("dcfsb")!=std::string::npos 
		|| mode.find("dc")!=std::string::npos  
		|| mode.find("grp")!=std::string::npos  
		|| mode.find("pipe")!=std::string::npos )
		return true;
	return false;
}

bool Option::processAlgorithm(){
	for(char& ch : algorighm){
		if(ch >= 'A' && ch <= 'Z')
			ch += 'a' - 'A';
	}
	if(algorighm == "lr"){
		return true;
	} else if(algorighm == "mlp"){
		return true;
	} else if(algorighm == "cnn"){
		return true;
	} else if(algorighm.find("km")!=std::string::npos){
		return true;
	} else if(algorighm.find("nmf")!=std::string::npos){
		return true;
	} else if(algorighm.find("lda")!=std::string::npos){
		return true;
	} else if(algorighm.find("gmm")!=std::string::npos){
		return true;
	}
	return false;
}
