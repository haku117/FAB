#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include "data/DataHolder.h"
#include "train/Trainer.h"
#include "train/GD.h"
#include "train/EM.h"
#include "util/Util.h"
#include "func.h"
#include "ParameterIO.h"

using namespace std;

double vectorDifference(const vector<double>& a, const vector<double>& b){
	double res = 0.0;
	size_t n = a.size();
	for(size_t i = 0; i < n; ++i){
		double t = a[i] - b[i];
		res += t * t;
	}
	return sqrt(res);
}

struct Option {
	string alg;
	string algParam;
	string batchSize;
	string fnRecord;
	string fnData;
	vector<int> idSkip;
	vector<int> idY;
	string fnParam;
	string fnOutput;
	bool doNormalize = true;
	bool show = false;

	bool parse(int argc, char* argv[]){
		int idx = 1;
		int optIdx = 7;
		if(argc <= 6)
			return false;
		try{
			alg = argv[idx++];
			algParam = argv[idx++];
			batchSize = argv[idx++];
			fnRecord = argv[idx++];
			fnData = argv[idx++];
			idSkip = getIntList(argv[idx++]);
			idY = getIntList(argv[idx++]);
			if(argc > optIdx++){
				fnParam = argv[idx++];
				processFn(fnParam);
			}
			if(argc > optIdx++){
				fnOutput = argv[idx++];
				processFn(fnOutput);
			}
			if(argc > optIdx++)
				doNormalize = beTrueOption(argv[idx++]);
			if(argc > optIdx++)
				show = beTrueOption(argv[idx++]);
		} catch(exception& e){
			cerr << "Cannot parse the " << idx << "-th parameter: " << argv[idx] << endl;
			cerr << "Error message: " << e.what() << endl;
			return false;
		}
		return true;
	}
	void usage(){
		cout << "usage: <alg> <alg-param> <batch-size> <fn-record> <fn-data> <id-skip> <id-y> [fn-param]"
			" [fn-output] [normalize=true] [show=false]" << endl
			<< "  <fn-record> and <fn-data> are required.\n"
			<< "  [fn-param] and [fn-output] can be omitted or given as '-'\n"
			<< endl;
	}
	void processFn(string& fn){
		if(fn == "-" || fn == " ")
			fn.clear();
	}
};

struct ParameterLoader{
	pair<string, vector<double>> loadParameter(const string& name, ifstream& fin){
		if(name == "lr")
			return funLR(fin);
		else
			return funGeneral(fin);
		return {};
	}
private:
	pair<string, vector<double>> funLR(ifstream& fin){
		string line;
		getline(fin, line);
		vector<double> vec = getDoubleList(line);
		string param = to_string(vec.size() - 1);
		return make_pair(move(param), move(vec));
	}
	pair<string, vector<double>> funGeneral(ifstream& fin){
		string line;
		getline(fin, line);
		string param = line;
		vector<int> shape = getIntList(line, " ,-");
		vector<double> vec;
		for(size_t i = 0; i < shape.size() - 1; ++i){
			getline(fin, line);
			vector<double> temp = getDoubleList(line);
			vec.insert(vec.end(), temp.begin(), temp.end());
		}
		return make_pair(move(param), move(vec));
	}
};

int main(int argc, char* argv[]){
	Option opt;
	if(!opt.parse(argc, argv)){
		opt.usage();
		return 1;
	}
	ios_base::sync_with_stdio(false);

	ifstream fin(opt.fnRecord);
	if(fin.fail()){
		cerr << "cannot open record file: " << opt.fnRecord << endl;
		return 4;
	}
	string algParam = opt.algParam;
	vector<double> ref;
	const bool withRef = !opt.fnParam.empty();
	if(withRef){
		ifstream finr(opt.fnParam);
		if(finr.fail()){
			cerr << "cannot open reference parameter file: " << opt.fnParam << endl;
			return 5;
		}
		ParameterIO io(opt.alg, "");
		string tmp;
		tie(tmp, ref) = io.load(finr);
		if(tmp != algParam){
			cerr << "Warning: given parameter does not match the one read from parameter file" << endl;
		}
		algParam = tmp;
	}

	ofstream fout(opt.fnOutput);
	if(!opt.fnOutput.empty() && fout.fail()){
		cerr << "cannot open output file: " << opt.fnOutput << endl;
		return 2;
	}
	const bool write = !opt.fnOutput.empty();
	int pp = 1;
	if (opt.alg == "km") pp = 17;
	else if (opt.alg == "lr") pp = 17;
	else if (opt.alg == "mlp") pp = 59;

	DataHolder dh(false, 1, 0);
	if(opt.alg.find("nmf") !=std::string::npos)
		dh.loadNMF(opt.fnData, ",", opt.algParam, false, true);
	else if(opt.alg.find("lda") !=std::string::npos)
		dh.loadLDA(opt.fnData, ",", 7);
	else
		dh.load(opt.fnData, ",", opt.idSkip, opt.idY, true, false, pp);

	// if(opt.doNormalize)
	// 	dh.normalize(false);

	Model m;
	try{
		m.init(opt.alg, dh.xlength(), algParam);
	} catch(exception& e){
		cerr << "Error in initialize model" << endl;
		cerr << e.what() << endl;
		return 3;
	}

	Parameter param;
	Trainer* trainer;
	if (opt.alg == "km" || opt.alg == "nmf" || opt.alg == "lda") {
		trainer = new EM;
	}else {
		trainer = new GD;
	}
	trainer->bindDataset(&dh);
	trainer->bindModel(&m);

	if(opt.show)
		cout << "finish binding " << m.paramWidth() << endl;

	string line;
	vector<double> last;
	double lastLoss = 0;
	//int idx = 0;
	int count = 0;
	int tokenSize = -1;
	while(getline(fin, line)){
		if(line.size() < 3)
			continue;
		//if(idx++ < 500)
		//	continue;
		// if(opt.show)
		// 	cout << "parse line: " << line << endl;

		pair<vector<string>, vector<double>> p;
		// if(count++ == 0 && opt.alg != "sync")
		// 	p = parseRecordLineIter(line, false);
		// else
			p = parseRecordLineIter(line, opt.doNormalize);
			count++;

		if (tokenSize == -1){
			tokenSize = p.second.size();
		}
		else if (tokenSize != p.second.size()){
			cout << "token Size prob: " << tokenSize << " -- " << p.second.size();
			continue; // assert the same param size
		}
		// if(opt.show)
		// 	cout << "time size: " << p.first.size() << "\tparam size: " << p.second.size() << endl;
		double diff = 0.0;
		// if(withRef)
		// 	diff = vectorDifference(ref, p.second);
		// if(last.empty()){
		// 	last = vector<double>(p.second.size(), 0.0);
		// }
		// double impro = vectorDifference(last, p.second);
		
		// if(opt.show)
		// 	cout << "improv " << impro << endl;
		
		last = p.second;
		param.set(move(p.second));
		m.setParameter(move(param));

		double loss = lastLoss;
		// if (impro > 0.0001) {
			loss = trainer->loss();
		double impro = loss - lastLoss;
			lastLoss = loss;
		// }

		if(opt.show) {
			if(opt.doNormalize)
				cout << p.first[1] << "," << loss << "," << impro << "," << p.first[2] 
					<< "," << p.first[3] << "," << p.first[0] << endl;
			else
				cout << p.first[1] << "\t" << loss << "\t" << impro << "\t" << p.first[0] << endl;
		}
		if(write) {
			if(opt.doNormalize) // or opt.alg == "sync")
				fout << p.first[1] << "," << loss << "," << impro << "," << p.first[2] 
					// << "," << (stoi(p.first[0])*stoi(opt.batchSize)) << "," << p.first[0] << "\n";
					<< "," << p.first[3] << "," << p.first[0] << "\n";
			else
				fout << p.first[1] << "," << loss << "," << impro << "," << p.first[0] 
					<< "," << p.first[3] << "\n";
			fout.flush();
		}
	}
	fin.close();
	fout.close();
	return 0;
}
