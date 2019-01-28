#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include "data/DataHolder.h"
#include "train/GD.h"
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
		cout << "usage: <alg> <alg-param> <fn-record> <fn-data> <id-skip> <id-y> [fn-param]"
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
		else if(name == "mlp")
			return funMLP(fin);
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
	pair<string, vector<double>> funMLP(ifstream& fin){
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
			cerr << "Warning: given parameter does not match the one read from file" << endl;
		}
		algParam = tmp;
	}

	ofstream fout(opt.fnOutput);
	if(!opt.fnOutput.empty() && fout.fail()){
		cerr << "cannot open output file: " << opt.fnOutput << endl;
		return 2;
	}
	const bool write = !opt.fnOutput.empty();

	DataHolder dh(false, 1, 0);
	dh.load(opt.fnData, ",", opt.idSkip, opt.idY, false, true);
	if(opt.doNormalize)
		dh.normalize(false);

	Model m;
	try{
		m.init(opt.alg, dh.xlength(), algParam);
	} catch(exception& e){
		cerr << "Error in initialize model" << endl;
		cerr << e.what() << endl;
		return 3;
	}

	Parameter param;
	GD trainer;
	trainer.bindDataset(&dh);
	trainer.bindModel(&m);

	string line;
	vector<double> last(dh.xlength(), 0.0);
	//int idx = 0;
	while(getline(fin, line)){
		if(line.size() < 3)
			continue;
		//if(idx++ < 500)
		//	continue;
		pair<double, vector<double>> p = parseRecordLine(line);
		double diff = 0.0;
		if(withRef)
			diff = vectorDifference(ref, p.second);
		double impro = vectorDifference(last, p.second);
		last = p.second;
		param.set(move(p.second));
		m.setParameter(move(param));
		double loss = trainer.loss();
		if(opt.show)
			cout << p.first << "\t" << loss << "\t" << diff << "\t" << impro << endl;
		if(write)
			fout << p.first << "," << loss << "," << diff << "," << impro << "\n";
	}
	fin.close();
	fout.close();
	return 0;
}
