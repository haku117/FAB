#include "GD.h"
#include "logging/logging.h"
#include "util/Timer.h"
#include "util/Util.h"
using namespace std;

GD::GD()
	: rate(1.0)
{
}

void GD::setRate(const double rate) {
	if(rate >= 0)
		this->rate = rate;
	else
		this->rate = -rate;
}

double GD::getRate() const {
	return rate;
}

std::vector<double> GD::batchDelta(const size_t start, const size_t cnt, const bool avg)
{
	Timer tmr;
	int counter = 0;
	size_t end = start + cnt;
	if(end > pd->size())
		end = pd->size();
	size_t nx = pm->paramWidth();
	vector<double> grad(nx, 0.0);
	for(size_t i = start; i < end; ++i){
		VLOG(5) << "cal grad for i " << i;
		auto g = pm->gradient(pd->get(i));
		for(size_t j = 0; j < nx; ++j)
			grad[j] += g[j];
		counter++;
	}
	if(start != end && this->pm->kernelName() != "km"){
		// this is gradient DESCENT, so rate is set to negative
		double factor = -rate;
		if(avg)
			factor /= (end - start);
		for(auto& v : grad)
			v *= factor;
	}
	VLOG(2) << " unit dp cal time for " << counter << " : " << tmr.elapseSd()/counter;
	return grad;
}

std::pair<size_t, std::vector<double>> GD::batchDelta(
	std::atomic<bool>& cond, const size_t start, const size_t cnt, const bool avg)
{
	Timer tmr;
	int counter = 0;
	size_t end = start + cnt;
	if(end > pd->size())
		end = pd->size();
	size_t nx = pm->paramWidth();
	vector<double> grad(nx, 0.0);
	size_t i;

	for(i = start; i < end && (cond.load() || i == start); ++i){
		// auto g = pm->gradient(pd->get(i));
		// vector<int>* zi = &(z.at(i));
		auto g = pm->gradient(pd->get(i));
		for(size_t j = 0; j < nx; ++j)
			grad[j] += g[j];

		counter++;
	}
	if(i != start){
		// this is gradient DESCENT, so rate is set to negative
		double factor = -rate;
		if(avg)
			factor /= (i - start);
		for(auto& v : grad)
			v *= factor;
	}

	VLOG(2) << " unit dp cal time for " << counter << " : " << tmr.elapseSd()/counter;
	return make_pair(i - start, move(grad));
}

std::pair<size_t, std::vector<double>> GD::batchDelta(std::atomic<bool>& cond, 
	const size_t start, const size_t cnt, const double slp, const double interval, const bool avg)
{
	Timer tmr;
	int counter = 0;
	size_t end = start + cnt;
	if(end > pd->size())
		end = pd->size();
	size_t nx = pm->paramWidth();
	vector<double> grad(nx, 0.0);
	size_t i;

	for(i = start; i < end && (cond.load() || i == start) && tmr.elapseSd() < interval; ++i){
		// auto g = pm->gradient(pd->get(i));
		// vector<int>* zi = &(z.at(i));
		Timer tmr;
		auto g = pm->gradient(pd->get(i));
		for(size_t j = 0; j < nx; ++j)
			grad[j] += g[j];

		counter++;
		if (slp > 0.01){
			double base = tmr.elapseSd();
			sleep(base * slp);
		}
	}
	if(i != start){
		// this is gradient DESCENT, so rate is set to negative
		double factor = -rate;
		if(avg)
			factor /= (i - start);
		for(auto& v : grad)
			v *= factor;
	}

	VLOG(2) << " unit dp cal time for " << counter << " : " << tmr.elapseSd()/counter;
	return make_pair(i - start, move(grad));
}