#pragma once
#include "Kernel.h"
#include <vector>

class LDA
	: public Kernel
{
public:
	double alpha;
	// std::vector<double> beta; // apply to parameter weights??
	size_t num_topics;
	size_t num_terms;

	std::vector<double> oldSS;

	void init(const int xlength, const std::string& param);
	std::string name() const;
	bool dataNeedConstant() const;
	int lengthParameter() const;

	std::vector<double> predict(const std::vector<double>& x, const std::vector<double>& w,
		const std::vector<double>& y = std::vector<double>()) const;
	int classify(const double p) const;

	double loss(const std::vector<double>& pred, const std::vector<double>& label) const;
	std::vector<double> gradient(const std::vector<double>& x, const std::vector<double>& w, 
		const std::vector<double>& y, std::vector<double>* z = nullptr) const;

	// void updateLocalZ(std::vector<double>& zz);
	// std::vector<double> computeDelta(std::vector<double>& zz);
	
};
