#include "DataHolder.h"
#include "Loader.h"
#include "math.h"
#include "util/Util.h"
#include <algorithm>
#include <unordered_set>
#include <fstream>
#include "logging/logging.h"

using namespace std;

DataHolder::DataHolder(const bool appOne, const size_t nparts, const size_t localid)
	: appendOne(appOne), npart(nparts), pid(localid)
{}

size_t DataHolder::xlength() const{
	return nx + (appendOne ? 1 : 0);
}

size_t DataHolder::ylength() const{
	return ny;
}
//
void DataHolder::load(const std::string& fpath, const std::string& sepper,
	const std::vector<int> skips, const std::vector<int>& yIds,
	const bool header, const bool onlyLocalPart)
{
	ifstream fin(fpath);
	if(fin.fail()){
		throw invalid_argument("Error in reading file: " + fpath);
	}
	// calculate number of x
	nx = 0;
	int n = 0;
	string line;
	getline(fin, line);
	size_t p = line.find(sepper);
	while(p!=string::npos){
		++n;
		p = line.find(sepper, p+1);
	}
	++n;
	// calculate xIds and yId
	unordered_set<int> xIds;
	unordered_set<int> yIds_u;
	for(int i=0; i<n; ++i){
		if(find(skips.begin(), skips.end(), i) != skips.end())
			continue;
		if(find(yIds.begin(), yIds.end(), i) != yIds.end())
			yIds_u.insert(i);
		else
			xIds.insert(i);
	}
	//nx = n - skips.size() - 1; // does not work when skips contains placeholders like -1
	nx = xIds.size();
	ny = yIds_u.size();

	// deal with header
	if(!header)
		fin.seekg(0);
	// parse lines
	size_t lid = 0; // line id;
	while(getline(fin, line)){
		if(line.size() < nx || line.front() == '#') // invalid line
			continue;
		if(onlyLocalPart && lid++ % npart != pid) // not local line
			continue;
		DataPoint dp = parseLine(line, sepper, xIds, yIds_u, appendOne);
		data.push_back(move(dp));
	}
}

void DataHolder::loadNMF(const std::string& fpath, const std::string& sepper,
	// const std::vector<int> skips, const std::vector<int>& yIds,
	const std::string& param,
	const bool header, const bool onlyLocalPart)
{
	ifstream fin(fpath);
	if(fin.fail()){
		throw invalid_argument("Error in reading file: " + fpath);
	}
	// LOG(INFO) << "start loading NMF data: " << nnx << ", " << nny;
	// calculate number of x
	nx = 0;
	int xi = -1; // x index
	string line;

	std::vector<int> params = parseParam(param);
	int nnx = params[1];
	int nny = params[2];

	// int sqrtNW = sqrt(npart);
	// int xparts = sqrtNW;
	// int yparts = sqrtNW;
	// if(sqrtNW * sqrtNW < npart && sqrtNW * (sqrtNW + 1) >= npart) {
	// 	xparts = sqrtNW+1;
	// }
	// else if(sqrtNW * (sqrtNW + 1) < npart) {
	// 	xparts = sqrtNW + 1;
	// 	yparts = sqrtNW + 1;
	// }

	// deal with header
	if(!header)
		fin.seekg(0);
	// parse lines
	size_t lid = 0; // line id;
	while(getline(fin, line)){
		if(line.front() == '#') // invalid line
			continue;
		++xi;

		if(onlyLocalPart && lid++ % npart != pid) // not local line
			continue;
		parseLineNMF(line, sepper, data, xi, appendOne);
		// DataPoint dp = parseLineNMF(line, sepper, xIds, yIds_u, appendOne);
		// data.push_back(move(dp));
		// if(xi % 100 == 1){
		// 	VLOG(2) << "load NMF data: " << xi;
		// }
	}
}

void DataHolder::add(const std::vector<double>& x, const std::vector<double>& y){
	nx = x.size();
	ny = y.size();
	DataPoint dp;
	dp.x = x;
	dp.y = y;
	data.push_back(move(dp));
}

void DataHolder::add(std::vector<double>&& x, std::vector<double>&& y){
	nx = x.size();
	ny = y.size();
	DataPoint dp;
	dp.x = move(x);
	dp.y = move(y);
	data.push_back(move(dp));
}

// normalize to [-1, 1]
void DataHolder::normalize(const bool onY)
{
	if(data.size() < 2)
		return;
	vector<double> max_x = data.front().x;
	vector<double> min_x = data.front().x;
	vector<double> max_y = data.front().y;
	vector<double> min_y = data.front().y;
	for(const auto& d : data){
		for(size_t i = 0; i < nx; ++i){
			if(d.x[i] > max_x[i])
				max_x[i] = d.x[i];
			else if(d.x[i] < min_x[i])
				min_x[i] = d.x[i];
		}
		if(onY){
			if(d.y > max_y)
				max_y = d.y;
			else if(d.y < min_y)
				min_y = d.y;
		}
	}
	vector<double> range_x(nx);
	vector<double> range_y(ny);
	for(size_t i = 0; i < nx; ++i)
		range_x[i] = max_x[i] - min_x[i];
	for(size_t i = 0; i < ny; ++i)
		range_y[i] = max_y[i] - min_y[i];
	for(auto&d : data){
		for(size_t i = 0; i < nx; ++i)
			d.x[i] = 2 * (d.x[i] - min_x[i]) / range_x[i] - 1;
		if(onY){
			for(size_t i = 0; i < ny; ++i)
				d.y[i] = 2 * (d.y[i] - min_y[i]) / range_y[i] - 1;
		}
	}
}
