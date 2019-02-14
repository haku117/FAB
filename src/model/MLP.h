#pragma once
#include "Kernel.h"
#include "MLPProxy.h"

class MLP
	: public Kernel
{
	int nLayer;
	std::vector<int> nNodeLayer;
	mutable MLPProxy proxy; // non-const bind function required
public:
	void init(const int xlength, const std::string& param);
	std::string name() const;
	bool dataNeedConstant() const;
	int lengthParameter() const;

	std::vector<double> predict(const std::vector<double>& x, const std::vector<double>& w) const;
	int classify(const double p) const;

	double loss(const std::vector<double>& pred, const std::vector<double>& label) const;
	// std::vector<double> gradient(
	// 	const std::vector<double>& x, const std::vector<double>& w, const std::vector<double>& y) const;
	std::vector<double> gradient(const std::vector<double>& x, const std::vector<double>& w, 
		const std::vector<double>& y, std::vector<int>* z = nullptr) const;

private:
	double getWeight(const std::vector<double>& w, const int layer, const int from, const int to) const;

	// require: proxy.bind(&w) had been called before
	std::vector<double> activateLayer(
		const std::vector<double>& x, const std::vector<double>& w, const int layer) const;
};
