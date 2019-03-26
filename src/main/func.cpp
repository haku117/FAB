#include "func.h"

using namespace std;

pair<double, vector<double>> parseRecordLine(const string& line){
	size_t pl = 0;
	size_t p = line.find(',');
	int id = stoi(line.substr(pl, p - pl)); // iteration-number
	pl = p + 1;
	p = line.find(',', pl);
	double time = stod(line.substr(pl, p - pl)); // time
	pl = p + 1;
	p = line.find(',', pl);
	vector<double> weights;
	// weights
	while(p != string::npos){
		weights.push_back(stod(line.substr(pl, p - pl)));
		pl = p + 1;
		p = line.find(',', pl);
	}
	weights.push_back(stod(line.substr(pl)));
	return make_pair(move(time), move(weights));
}

pair<vector<double>, vector<double>> parseRecordLineIter(const string& line){
	size_t pl = 0;
	size_t p = line.find(',');
	int id = stoi(line.substr(pl, p - pl)); // iteration-number
	pl = p + 1;
	p = line.find(',', pl);
	double time = stod(line.substr(pl, p - pl)); // time
	pl = p + 1;
	p = line.find(',', pl);
	vector<double> weights;
	// weights
	while(p != string::npos){
		weights.push_back(stod(line.substr(pl, p - pl)));
		pl = p + 1;
		p = line.find(',', pl);
	}
	weights.push_back(stod(line.substr(pl)));
	vector<double> idT;
	idT.push_back((double)id);
	idT.push_back(time);
	return make_pair(move(idT), move(weights));
}
