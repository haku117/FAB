#include "Parameter.h"
#include <random>
using namespace std;

void Parameter::init(const std::vector<double>& w)
{
	auto t = w;
	set(t);
}

void Parameter::init(const size_t n, const double v)
{
	this->n = n;
	weights.assign(n, v);
}

void Parameter::init(const size_t n, const double mu, const double sigma, const unsigned seed)
{
	normal_distribution<double> dist(mu, sigma);
	mt19937 g(seed);
	auto gen = [&](){ 
		double t;
		do{
			t = dist(g);
		} while(t == 0.0);
		return t;
	};
	init(n, gen);
}

void Parameter::init(const size_t n, std::function<double()> gen)
{
	this->n = n;
	for(size_t i = 0; i < n; ++i){
		double v = gen();
		weights.push_back(v);
	}
}

void Parameter::set(const std::vector<double>& d)
{
	n = d.size();
	weights = d;
}
void Parameter::set(std::vector<double>&& d)
{
	n = d.size();
	weights = std::move(d);
}

void Parameter::accumulate(const std::vector<double>& delta){
	for(size_t i=0; i<n; ++i)
		weights[i] += delta[i];
}

void Parameter::accumulate(const std::vector<double>& grad, const double rate){
	for(size_t i=0; i<n; ++i)
		weights[i] += rate*grad[i];
}

bool Parameter::isSameParm(const Parameter& pp){
	int psize = pp.size();
	if(psize != weights.size())
		return false;

	std::vector<double> ppw = pp.getWeights();
	for (int i = 0; i < psize; i++){
		if(ppw[i] != weights[i])
			return false;
	}
	return true;
}