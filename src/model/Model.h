#pragma once
#include "Parameter.h"
#include "Kernel.h"
#include "data/DataHolder.h"
#include <vector>
#include <string>

class Model {
	Parameter param;
	//LogisticRegression kern; // with be changed to a general interface
	Kernel* kern = nullptr;

public:
	// initialize kernel, do not initialize parameter
	void init(const std::string& name, const int nx, const std::string& paramKern);
	// initialize kernel, initialize parameter using <w0>
	void init(const std::string& name, const int nx, const std::string& paramKern, const double w0);
	// initialize kernel, initialize parameter using random value ~ N(0.01, 0.01)
	void init(const std::string& name, const int nx, const std::string& paramKern, const unsigned seed);
	// initialize kernel, initialize parameter using given param (for km??)
	void init(const std::string& name, const int nx, const std::string& paramKern,
			const std::vector<double>& pm);
	void clear();
	std::string kernelName() const;

	//void initParamWithData(const DataPoint& d);
	//void initParamWithSize(const size_t n);

	void setParameter(const Parameter& p);
	void setParameter(Parameter&& p);
	Parameter& getParameter();
	const Parameter& getParameter() const;
	size_t paramWidth() const;

	Kernel* getKernel();

	void accumulateParameter(const std::vector<double>& grad, const double factor);
	void accumulateParameter(const std::vector<double>& grad);
	
	std::vector<double> predict(const DataPoint& dp) const;
	int classify(const double p) const;
	double loss(const DataPoint& dp) const;
	double loss(const std::vector<double>& pred, const std::vector<double>& label) const;
	// std::vector<double> gradient(const DataPoint& dp) const;
	std::vector<double> gradient(const DataPoint& dp, std::vector<int>* z) const;

private:
	void generateKernel(const std::string& name);
};
