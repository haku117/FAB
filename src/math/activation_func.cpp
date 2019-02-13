#include "activation_func.h"
#include <cmath>
using namespace std;

double sigmoid(double x) {
	return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double x)
{
	double t = sigmoid(x);
	return t * (1.0 - t);
}

double sigmoid_derivative(double x, double y)
{
	return y * (1.0 - y);
}

double relu(double x){
	return x >= 0.0 ? x : 0;
}

double relu_derivative(double x){
	return x >=0.0 ? 1 : 0;
}

double relu_derivative(double x, double y)
{
	return x >= 0.0 ? 1 : 0;
}

double tanh_derivative(double x){
	double t = tanh(x);
	return 1.0 - t*t;
}

double tanh_derivative(double x, double y)
{
	return 1.0 - y * y;
}

double arctan(double x){
	return atan(x);
}

double arctan_derivative(double x){
	return 1.0 / (x*x + 1);
}

double arctan_derivative(double x, double y)
{
	return 1.0 / (x*x + 1);
}

double softplus(double x){
	return log1p(exp(x));
}

double softplus_derivative(double x){
	return sigmoid(x);
}

double softplus_derivative(double x, double y)
{
	return sigmoid(x);
}

std::vector<double> softmax(const std::vector<double>& x){
	std::vector<double> res(x.size());
	double sum=0.0;
	for(size_t i=0;i<x.size();++i){
		double t = exp(x[i]);
		res[i]=t;
		sum+=t;
	}
	for(double& v : res){
		v/=sum;
	}
	return res;
}

std::vector<std::vector<double>> softmax_derivative(const std::vector<double>& x){
	const size_t n = x.size();
	std::vector<std::vector<double>> res(n, vector<double>(n));
	for(size_t i=0;i<n;++i){
		for(size_t j=0;j<n;++j)
			if(i==j)
				res[i][j]=x[i]*(1-x[j]);
			else
				res[i][j]=-x[i]*x[j];
	}
	return res;
}
