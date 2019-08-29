#pragma once
#include <vector>
#include <functional>

struct Parameter {
	std::vector<double> weights;
	size_t n;

public:
	// initialize with a given vector
	void init(const std::vector<double>& w);
	// initialize with fixed number
	void init(const size_t n, const double v);
	// initialize with normal distribution
	void init(const size_t n, const double mu, const double sigma2, const unsigned seed = 0);
	// initialize with a value generator
	void init(const size_t n, std::function<double()> gen);
	
	void initLDA(const size_t n, const int k, const unsigned seed);

	void set(const std::vector<double>& d);
	void set(std::vector<double>&& d);
	void reset(const double v);
	size_t size() const { return weights.size(); }

	std::vector<double> getWeights() const { return weights; }

	std::vector<double> getLDAweights(bool flag = false);
	std::vector<double> getGMMweights(int K, int D, int cnt, bool flag = false);

	bool isSameParm(const Parameter& pp);
	
	void accumulate(const std::vector<double>& delta);
	void accumulate(const std::vector<double>& grad, const double rate);

	// void accumulateLDA(const std::vector<double>& ss, int k);
};
