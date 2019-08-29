#include <iostream>
#include <fstream>
#include <random>
#include <string>
#include <vector>
#include <functional>
#include "model/KernelFactory.h"
#include "model/Kernel.h"
#include "util/Util.h"
#include "func.h"
#include "ParameterIO.h"

using namespace std;

struct Option{
	string fnameR, fnameP; // filename of record and parameter
	string algorithm;
	int xlength;
	int ylength;
	int n;
	string param;
	double paraMin = -10.0;
	double paraMax = 10.0;
	double errSigma = 0.1;
	unsigned long seed = 123;

	bool parse(int argc, char* argv[]){
		int idx = 1;
		int optIdx = 8;
		if(argc <= 7)
			return false;
		try{
			fnameR = argv[idx++];
			fnameP = argv[idx++];
			algorithm = argv[idx++];
			param = argv[idx++];
			xlength = stoi(argv[idx++]);
			ylength = stoi(argv[idx++]);
			n = stoiKMG(argv[idx++]);
			if(argc > optIdx++)
				paraMin = stod(argv[idx++]);
			if(argc > optIdx++)
				paraMax = stod(argv[idx++]);
			if(argc > optIdx++)
				errSigma = stod(argv[idx++]);
			if(argc > optIdx++)
				seed = stoul(argv[idx++]);
		} catch(exception& e){
			cerr << "Cannot parse the " << idx << "-th parameter: " << argv[idx] << endl;
			cerr << "Error message: " << e.what() << endl;
			return false;
		}
		if(!formateAlg()){
			cerr << "Error in checking algorithm and its parameters" << endl;
			return false;
		}
		return true;
	}
	void usage() const{
		cout << "Usage: <fname-r> <fname-p> <alg> <alg-param> <xlength> <ylength> <n> [pmin=-1] [pmax=1] [err=0.1] [seed=123]\n"
			<< "Generate <x'> and <y> where y=alg(x, param), <param> is auto generated and x'=x+err.\n"
			<< "Output the auto generated parameters in standared output.\n"
			<< "The range of each entry of <x> is [-1,1]. Error follows normal distribution N(0, <err>^2).\n"
			<< "Example: generate record.txt \"\" lr 3 3 1 100 -1 1 0.05 123456\n"
			<< "  <fname-r> <fname-p>: file name of generated records and parameter. If <fnname-p> is empty, it is omitted.\n"
			<< "  <alg>: supports: LR (Logistic Regession).\n"
			<< "  <alg-param>: the parameters for the algorithm, usually the shape of the model.\n"
			<< "  <xlength> <ylength>: length of the x/y part of each generated record.\n"
			<< "  <n>: number of lines of the output record file, supports k,m,g suffix.\n"
			<< "  [pmin] [pmax]: the minimum and maximum of the model parameters.\n"
			<< "  [err]: the standard derivation (sigma) of error on each data point.\n";
	}
private:
	bool formateAlg(){
		for(char& ch : algorithm){
			if(ch >= 'A' && ch <= 'Z')
				ch += 'a' - 'A';
		}
		// cout << "gen algorithm " << algorithm;
		if(algorithm == "lr"){
			return true;
		} else if(algorithm == "mlp"){
			vector<int> shape = getIntList(param, "_");
			cout << "mlp shape: " << shape.size() << "," << shape.front() << "," << shape.back() << endl;
			return shape.size() >= 2 && shape.front() == xlength && shape.back() == ylength;
		} else if(algorithm == "cnn"){
			vector<string> shape = getStringList(param, "-");
			return shape.size() >= 2;
		}
		return false;
	}
};

class ParameterGenerator{
	Kernel* k;
	int xlength;
	double pmin, pmax;
	string param;

	using fn_g_t = vector<double>(ParameterGenerator::*)(mt19937&);
	fn_g_t fpg;
	ParameterIO io;
public:
	ParameterGenerator(Kernel* k,
		const int xlength, const double pmin, const double pmax)
		: k(k), xlength(xlength), pmin(pmin), pmax(pmax), io(k->name(), k->parameter())
	{
		if(k->name() == "lr"){
			fpg = &ParameterGenerator::genLR;
		} else if(k->name() == "mlp"){
			fpg = &ParameterGenerator::genGeneral;
		} else if(k->name() == "cnn"){
			fpg = &ParameterGenerator::genGeneral;
		}
	}
	vector<double> gen(mt19937& gen){
		return (this->*fpg)(gen);
	}
	void write(ostream& os, vector<double>& p){
		return io.write(os, p);
	}
private:
	// an additional one. the last one is the constant offset
	vector<double> genLR(mt19937& gen){
		vector<double> res;
		uniform_real_distribution<double> dis(pmin, pmax);
		for(int i = 0; i < xlength + 1; ++i){
			res.push_back(dis(gen));
		}
		return res;
	}
	vector<double> genGeneral(mt19937& gen){
		int n = k->lengthParameter();
		vector<double> res;
		uniform_real_distribution<double> dis(pmin, pmax);
		for(int i = 0; i < n; ++i){
			res.push_back(dis(gen));
		}
		return res;
	}
};


class Dumper{
	Kernel* k;
	ofstream fout;
	const vector<double>* param;
	const size_t xlength;
	const size_t ylength;
	uniform_real_distribution<double> xdis; // x-data
	function<double(mt19937&)> egen; // error
	normal_distribution<double> edis; // error

	using fn_t = void (Dumper::*)(mt19937&);
	fn_t fp;
public:
	Dumper(Kernel* k, const string& fname, const vector<double>* pparam,
		const size_t xlength, const size_t ylength, const double sigma)
		: k(k), fout(fname), param(pparam),
		xlength(xlength), ylength(ylength), xdis(-1, 1)
	{
		if(fout.fail()){
			cerr << "Cannot write to file: " << fname << endl;
		}
		if(k->name() == "lr"){
			fp = &Dumper::dumpClassify;
		} else if(k->name() == "mlp"){
			fp = &Dumper::dumpClassify;
		} else if(k->name() == "cnn"){
			fp = &Dumper::dumpClassify;
		}
		if(sigma != 0.0){
			edis = normal_distribution<double>(0, sigma);
			egen = [&](mt19937& gen){
				return edis(gen);
			};
		} else{
			egen = [](mt19937&){return 0; };
		}
	}
	void dumpLine(mt19937& gen){
		(this->*fp)(gen);
	}
private:
	void dumpClassify(mt19937& gen){
		vector<double> x(xlength, 0.0);
		// generate x
		for(size_t i = 0; i < xlength; ++i)
			x[i] = xdis(gen);
		// calcualte y
		vector<double> y = k->predict(x, *param);
		vector<int> yy(ylength);
		for(size_t i = 0; i < ylength; ++i)
			yy[i] = k->classify(y[i]);
		y.clear();
		// add error to x
		for(size_t i = 0; i < xlength; ++i)
			x[i] += edis(gen);
		// dump
		for(size_t i = 0; i < xlength; ++i)
			fout << x[i] << ",";
		for(size_t i = 0; i < ylength - 1; ++i)
			fout << yy[i] << ",";
		fout << yy.back() << "\n";
	}
};

int main(int argc, char* argv[]){
	Option opt;
	if(!opt.parse(argc, argv)){
		opt.usage();
		return 1;
	}
	ios_base::sync_with_stdio(false);
	mt19937 gen(opt.seed);

	try{
		Kernel* k = KernelFactory::generate(opt.algorithm);
		k->init(opt.xlength, opt.param);
		ParameterGenerator pg(k, opt.xlength, opt.paraMin, opt.paraMax);
		vector<double> param = pg.gen(gen);
		ofstream foutP(opt.fnameP);
		bool flag = foutP.is_open();
		cout << "Generated Parameters:" << endl;
		pg.write(cout, param);
		if(flag){
			pg.write(foutP, param);
			foutP.close();
		}

		Dumper dumper(k, opt.fnameR, &param, opt.xlength, opt.ylength, opt.errSigma);
		for(int i = 0; i < opt.n; ++i){
			dumper.dumpLine(gen);
		}
	} catch(exception& e){
		cerr << e.what() << endl;
		return 2;
	}

	return 0;
}
