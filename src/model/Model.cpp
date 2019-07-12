#include "Model.h"
#include "KernelFactory.h"
// #include <iostream>

using namespace std;

/*
void Model::initParamWithData(const DataPoint& d){
	param.init(d.x.size(), 0.01);
}

void Model::initParamWithSize(const size_t n){
	param.init(n, 0.01);
}*/

void Model::init(const std::string& name, const int nx, const std::string & paramKern)
{
	generateKernel(name);
	kern->init(nx, paramKern);
	
	// if (name.find("lda") != string::npos){
	// 	size_t nparam = kern->lengthParameter(); 
	// 	param.initLDA(nparam, stoi(paramKern), unsigned(1)); // nparam: ss size
	// }
}

void Model::init(const std::string& name, const int nx, const std::string & paramKern, const double w0)
{
	generateKernel(name);
	kern->init(nx, paramKern);
	size_t n = kern->lengthParameter();
	param.init(n, w0);
}

void Model::init(const std::string & name, const int nx, const std::string & paramKern, const unsigned seed)
{
	generateKernel(name);
	kern->init(nx, paramKern);
	size_t nparam = kern->lengthParameter(); 
	if (name.find("lda") != string::npos){
		param.initLDA(nparam, stoi(paramKern), seed); // nparam: ss size
	}
	else
		param.init(nparam, 1, 1, seed);
}

/// for km or specific param initalization
void Model::init(const std::string & name, const int nx, const std::string & paramKern, 
				const std::vector<double>& pm)
{
	generateKernel(name);
	kern->init(nx, paramKern);
	param.init(pm);
}

void Model::clear()
{
	delete kern;
	kern = nullptr;
}

std::string Model::kernelName() const{
	return kern->name();
}

void Model::setParameter(const Parameter& p) {
	param = p;
}
void Model::setParameter(Parameter&& p) {
	param = move(p);
}
void Model::resetparam(){
	param.reset(0);
}

Parameter & Model::getParameter()
{
	return param;
}

const Parameter & Model::getParameter() const
{
	return param;
}

size_t Model::paramWidth() const
{
	return param.size();
	// return kern->lengthParameter();
}

Kernel* Model::getKernel(){
	return kern;
}

void Model::accumulateParameter(const std::vector<double>& grad, const double factor)
{
	param.accumulate(grad, factor);
}

void Model::accumulateParameter(const std::vector<double>& grad)
{
	param.accumulate(grad);
}

// void Model::accumulateParameterLDA(const std::vector<double>& grad, int k)
// {
// 	param.accumulateLDA(grad, k);
// }

std::vector<double> Model::predict(const DataPoint& dp) const
{
	return kern->predict(dp.x, param.weights);
}

int Model::classify(const double p) const
{
	return kern->classify(p);
}

double Model::loss(const DataPoint & dp) const
{
	// cout << "start model loss" << endl;
	std::vector<double> pred = kern->predict(dp.x, param.weights, dp.y);
	// cout << "pred: " << pred.size() << "\ty: " << dp.y.size() << endl;
	return loss(pred, dp.y);
}

double Model::loss(const std::vector<double>& pred, const std::vector<double>& label) const
{
	return kern->loss(pred, label);
}

std::vector<double> Model::gradient(const DataPoint & dp, std::vector<int>* z)
{
	return kern->gradient(dp.x, param.weights, dp.y, z);
}

void Model::generateKernel(const std::string & name)
{
	if(kern != nullptr){
		delete kern;
		// kern = nullptr;
	}
	kern = KernelFactory::generate(name);
}

void Model::updateLocalZ(std::vector<double>& zz){
	kern->updateLocalZ(zz);
}

std::vector<double> Model::computeDelta(std::vector<double>& zz){
	return kern->computeDelta(zz);
}
