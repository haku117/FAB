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

pair<vector<std::string>, vector<double>> parseRecordLineIter(
		const string& line, const bool staleStats){
	size_t pl = 0;
	size_t p = line.find(',');
	std::string id = line.substr(pl, p - pl); // iteration-number
	pl = p + 1;
	p = line.find(',', pl);
	std::string time = line.substr(pl, p - pl); // time
	pl = p + 1;
	p = line.find(',', pl);
	std::string dp = line.substr(pl, p - pl); // time
	pl = p + 1;
	p = line.find(',', pl);

	std::string stale;	
	if(staleStats) {
		stale = line.substr(pl, p - pl); // stale info
		pl = p + 1;
		p = line.find(',', pl);
	}

	vector<double> weights;
	// weights
	while(p != string::npos){
		weights.push_back(stod(line.substr(pl, p - pl)));
		pl = p + 1;
		p = line.find(',', pl);
	}
	if(pl < line.size())
		weights.push_back(stod(line.substr(pl)));
	vector<std::string> idT;
	idT.push_back(id);
	idT.push_back(time);
	idT.push_back(stale);
	idT.push_back(dp);
	return make_pair(move(idT), move(weights));
}
