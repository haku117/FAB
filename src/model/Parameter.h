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

	void set(const std::vector<double>& d);
	void set(std::vector<double>&& d);
	std::vector<double> getWeights() const { return weights; }
	size_t size() const { return weights.size(); }

	bool isSameParm(const Parameter& pp);
	
	void accumulate(const std::vector<double>& delta);
	void accumulate(const std::vector<double>& grad, const double rate);
};
