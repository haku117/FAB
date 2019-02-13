#include "IntervalEstimator.h"
#include "math/norm.h"
#include <stdexcept>

using namespace std;

void IntervalEstimator::init(const size_t nWorker, const std::vector<std::string>& param)
{
	nw = nWorker;
	if (param[0] == "fixed") {
		fixedInterval = stod(param[1]);
		fp_update = &IntervalEstimator::mu_dummy;
		fp_update_vec = &IntervalEstimator::mu_dummy_vec;
		fp_interval = &IntervalEstimator::mi_fixed;
	}
	else {
		throw invalid_argument("Unspported interval estimator: " + param[0]);
	}
}

void IntervalEstimator::update(const double improve, const double interval, const double time)
{
	return (this->*fp_update)(improve, interval, time);
}

void IntervalEstimator::update(const std::vector<double>& improve, const double interval, const double time)
{
	return (this->*fp_update_vec)(improve, interval, time);
}

double IntervalEstimator::interval()
{
	return (this->*fp_interval)();
}


void IntervalEstimator::mu_dummy(const double improve, const double interval, const double time)
{}
void IntervalEstimator::mu_dummy_vec(const std::vector<double>& improve, const double interval, const double time)
{}
void IntervalEstimator::mu_dummy_vecl2(const std::vector<double>& improve, const double interval, const double time)
{
	return update(l2norm(improve), interval, time);
}
// ---- fixed ----

double IntervalEstimator::mi_fixed()
{
	return fixedInterval;
}


