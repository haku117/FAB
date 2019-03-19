#include "EM.h"
#include "util/Util.h"
#include "logging/logging.h"
using namespace std;

EM::EM()
	: rate(1.0) // always 1??
{
	// initState();
	// LOG(INFO) << "inital EM trainer ";
}

/// initialize the local state Z
void EM::initState(int dim){
	if(dim > 0) {
		int xlen = pd->size();
		// int dim = 1;//this.pm->getKernel()->lengthState();
		LOG(INFO) << "initalize local state Z: " << xlen << " " << dim;
		std::vector<std::vector<int> > matrix(xlen, std::vector<int>(dim, -1));
		z = move(matrix);
	}
}

void EM::setRate(const double rate) {
	this->rate = rate;
}

double EM::getRate() const {
	return rate;
}

// void EM::setZ(std::vector<std::vector<int> > z) {
// 	this->z = z;
// }

// std::vector<std::vector<int> > EM::getZ() const{
// 	return this->z;
// }

std::vector<double> EM::batchDelta(const size_t start, const size_t cnt, const bool avg)
{
	size_t end = start + cnt;
	// if(end > pd->size())
	// 	end = pd->size();
	size_t nx = pm->paramWidth();
	vector<double> delta(nx, 0.0);

	for(size_t dp = start; dp < end; ++dp){
		size_t i = dp % pd->size(); 	// round the data set
		VLOG(5) << "cal grad for i " << i << " z " << z[i];
		auto g = pm->gradient(pd->get(i), &(z[i]));
		for(size_t j = 0; j < nx; ++j)
			delta[j] += g[j];
	}

	return delta;
}

std::pair<size_t, std::vector<double>> EM::batchDelta(
	std::atomic<bool>& cond, const size_t start, const size_t cnt, const bool avg)
{
	size_t end = start + cnt;

	if(this->pm->kernelName() == "nmf") {
		size_t nx = pm->paramWidth();
		vector<double> delta(nx, 0.0);
		// std::vector<double> delta; //for accumuteDeltaSave version
		size_t dp; // data point index
		for(dp = start; dp < end && (cond.load() || dp == start); ++dp){
			size_t i = dp % pd->size(); 	// round the data set
			std::vector<double> d = pm->gradient(pd->get(i));
			for(size_t j = 0; j < d.size(); ++j)
				delta[j] += rate * d[j];
			// accumuteDeltaSave(delta, d);
		}
		return make_pair(dp - start, move(delta));
	}

	size_t nx = pm->paramWidth();
	vector<double> delta(nx, 0.0);
	size_t dp; // data point index
	for(dp = start; dp < end && (cond.load() || dp == start); ++dp){

		size_t i = dp % pd->size(); 	// round the data set
		auto g = pm->gradient(pd->get(i), &(z[i]));
		for(size_t j = 0; j < nx; ++j)
			delta[j] += g[j];
	}

	return make_pair(dp - start, move(delta));
}

// void EM::accumuteDeltaSave(std::vector<double>& delta, std::vector<double>& d){

// 	int rank = d.size()/2 - 1;
// 	int indxWu = int(d[0]);
// 	int indxHi = int(d[rank]);
// 	for(size_t j = 0; j < delta.size(); j += rank+1){
// 		if(delta[j] == indxWu) {// match the dimension of delta
// 			for(int k = 0; k < rank; k++){
// 				delta[j+1 + k] += d[1 + k]; 
// 			}
// 			indxWu = -1;
// 		}
// 		else if(delta[j] == indxHi){
// 			for(int k = 0; k < rank; k++){
// 				delta[j+1 + k] += d[rank+2 + k]; 
// 			}
// 			indxHi = -1;
// 		}
// 	}
// 	if(indxWu != -1) {
// 		delta.insert(delta.end(), d.begin(), d.begin()+rank+1);
// 	}
// 	if(indxHi != -1) {
// 		delta.insert(delta.end(), d.begin()+rank+1, d.end());
// 	}
// }