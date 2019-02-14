#include "ParameterIO.h"
#include "util/Util.h"
#include "model/CNNProxy.h"
#include <stdexcept>
using namespace std;

ParameterIO::ParameterIO(const std::string & name, const std::string & param)
	: name(name), param(param)
{
	vector<string> supported = { "lr", "mlp", "cnn", "km"};
	for(char& ch : this->name){
		if(ch >= 'A' && ch <= 'Z')
			ch += 'a' - 'A';
	}
	//find(supported.begin(), supported.end(), this->name) != supported.end();
	bool flag = false;
	for(auto&v : supported){
		if(v == this->name){
			flag = true;
			break;
		}
	}
	if(!flag)
		throw invalid_argument("ParameterIO module does not algorithm: " + name);
}

void ParameterIO::write(std::ostream & os, const std::vector<double>& w)
{
	if(name == "lr"){
		return writeLR(os, w);
	} else if(name == "mlp"){
		return writeMLP(os, w);
	} else if(name == "cnn"){
		return writeCNN(os, w);
	} else if(name == "km"){
		return writeKM(os, w);
	}
}

std::pair<std::string, std::vector<double>> ParameterIO::load(std::istream & is)
{
	if(name == "lr"){
		return loadLR(is);
	} else if(name == "mlp"){
		return loadMLP(is);
	} else if(name == "cnn"){
		return loadCNN(is);
	} else if(name == "km"){
		return loadKM(is);
	}
	return std::pair<std::string, std::vector<double>>();
}

// -------- LR --------

void ParameterIO::writeLR(std::ostream & os, const std::vector<double>& w)
{
	for(size_t i = 0; i < w.size() - 1; ++i){
		os << param[i] << ",";
	}
	os << w.back() << endl;
}

std::pair<std::string, std::vector<double>> ParameterIO::loadLR(std::istream & is)
{
	string line;
	getline(is, line);
	vector<double> vec = getDoubleList(line);
	string param = to_string(vec.size() - 1);
	return make_pair(move(param), move(vec));
}

// -------- MLP --------

void ParameterIO::writeMLP(std::ostream & os, const std::vector<double>& w)
{
	os << param << "\n";
	vector<int> shape = getIntList(param, " ,-");
	vector<int> nWeightOffset(shape.size());
	for(size_t i = 1; i < shape.size(); ++i)
		nWeightOffset[i] = nWeightOffset[i - 1] + (shape[i - 1] + 1)*shape[i];
	for(size_t l = 0; l < nWeightOffset.size() - 1; ++l){
		for(int i = nWeightOffset[l]; i < nWeightOffset[l + 1] - 1; ++i){
			os << w[i] << ",";
		}
		os << w[nWeightOffset[l + 1] - 1] << "\n";
	}
	os.flush();
}

std::pair<std::string, std::vector<double>> ParameterIO::loadMLP(std::istream & is)
{
	string line;
	getline(is, line);
	string param = line;
	vector<int> shape = getIntList(line, " ,-");
	vector<double> vec;
	for(size_t i = 0; i < shape.size() - 1; ++i){
		getline(is, line);
		vector<double> temp = getDoubleList(line);
		vec.insert(vec.end(), temp.begin(), temp.end());
	}
	return make_pair(move(param), move(vec));
}

// -------- CNN --------

void ParameterIO::writeCNN(std::ostream & os, const std::vector<double>& w)
{
	os << param << "\n";
	CNNProxy p;
	p.init(param);
	vector<int> nWeightOffset = p.weightOffsetLayer;
	for(size_t l = 1; l < nWeightOffset.size() - 1; ++l){
		if(p.nWeightNode[l] == 0)
			continue;
		for(int i = nWeightOffset[l]; i < nWeightOffset[l + 1] - 1; ++i){
			os << w[i] << ",";
		}
		os << w[nWeightOffset[l + 1] - 1] << "\n";
	}
	os.flush();
}

std::pair<std::string, std::vector<double>> ParameterIO::loadCNN(std::istream & is)
{
	string line;
	getline(is, line);
	string param = line;
	CNNProxy p;
	p.init(param);
	vector<double> vec;
	for(size_t i = 0; i < p.nLayer; ++i){
		if(p.nWeightNode[i] == 0)
			continue;
		getline(is, line);
		vector<double> temp = getDoubleList(line);
		vec.insert(vec.end(), temp.begin(), temp.end());
	}
	return make_pair(move(param), move(vec));
}

// -------- KMeans -------- ???????

void ParameterIO::writeKM(std::ostream & os, const std::vector<double>& w)
{
	for(size_t i = 0; i < w.size() - 1; ++i){
		os << param[i] << ",";
	}
	os << w.back() << endl;
}

std::pair<std::string, std::vector<double>> ParameterIO::loadKM(std::istream & is)
{
	string line;
	getline(is, line);
	vector<double> vec = getDoubleList(line);
	string param = to_string(vec.size() - 1);
	return make_pair(move(param), move(vec));
}
