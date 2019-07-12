#pragma once
#include "Kernel.h"
#include <vector>

class NMF
	: public Kernel
{
public:
	int nnx, nny;
	double lambda;
	// std::vector<int> z;

	void init(const int xlength, const std::string& param);
	std::string name() const;
	bool dataNeedConstant() const;
	int lengthParameter() const;

	std::vector<double> predict(const std::vector<double>& x, const std::vector<double>& w,
		const std::vector<double>& y = std::vector<double>()) const;
	int classify(const double p) const;

	double loss(const std::vector<double>& pred, const std::vector<double>& label) const;
	std::vector<double> gradient(const std::vector<double>& x, const std::vector<double>& w, 
		const std::vector<double>& y, std::vector<int>* z = nullptr) const;

	// vector<vector<double>> multiVectorConvert(const std::vector<double>& w) const;
	// double eudist(const std::vector<double>& x, const std::vector<double>& c, const int num) const;
};
