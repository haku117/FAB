#pragma once
#include <vector>
#include <string>

struct IntervalEstimator {
	void init(const size_t nWorker, const std::vector<std::string>& param);
	// kernel function:
	void update(const double improve, const double interval, const double time); 
	double interval(); // get the time to wait

	// supporting function:
	void update(const std::vector<double>& improve, const double interval, const double time);

	void mu_dummy(const double improve, const double interval, const double time);
	void mu_dummy_vec(const std::vector<double>& improve, const double interval, const double time);
	void mu_dummy_vecl2(const std::vector<double>& improve, const double interval, const double time);

private: // state
	using fp_update_t = void (IntervalEstimator::*)(const double, const double, const double);
	fp_update_t fp_update;
	using fp_update_vec_t = void (IntervalEstimator::*)(const std::vector<double>&, const double, const double);
	fp_update_vec_t fp_update_vec;
	using fp_interval_t = double (IntervalEstimator::*)();
	fp_interval_t fp_interval;

	size_t nw;
	double last_improve;
	double last_interval;

private: // interval method
	double fixedInterval;
	double mi_fixed();
};
