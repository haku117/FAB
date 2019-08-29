#include "Parameter.h"
#include <random>
#include <iostream>
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
		return abs(t); /// abs for nmf
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
void Parameter::reset(const double v)
{
	weights.assign(weights.size(), v);
}

void Parameter::accumulate(const std::vector<double>& delta){
	for(size_t i=0; i<n; ++i)
		weights[i] += delta[i];
}

void Parameter::accumulate(const std::vector<double>& grad, const double rate){
	for(size_t i=0; i<weights.size(); ++i)
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

// void Parameter::accumulateLDA(const std::vector<double>& ss, int k){

// 	// int K = grad.size() - n;
// 	// int V = n / K;
// 	// for(size_t i = 0; i < n; ++i)
// 	// 	weights[i] = log(grad[i]) - log(grad[n/V]);
// 	int N = ss.size() - k;
// 	int V = ss.size() / k - 1;
// 	for(size_t i = 0; i < N; ++i)
// 		weights[i] = log(ss[i]) - log(ss[N + i/V]);
// }

void Parameter::initLDA(const size_t ssSize, const int k, const unsigned seed)
{
	this->n = k;
	int num_terms = ssSize / k - 1;
	vector<double> ttk;
	double r, ttsum;
	for(size_t i = 0; i < ssSize - k; ++i){
		r = ((double) rand() / (RAND_MAX)) + (double)1 / num_terms;
		weights.push_back(r);
		ttsum += r;
		if (i % num_terms == num_terms - 1){
			ttk.push_back(ttsum);
		}
	}
	weights.insert(weights.end(), ttk.begin(), ttk.end());
}

std::vector<double> Parameter::getLDAweights(bool flag){

	std::vector<double> beta;
	int N = weights.size() - n;
	int V = (weights.size() - n) / n;

	for(size_t i = 0; i < N; ++i){
		if (weights[i] > 0) {
			double x = log(weights[i]) - log(weights[N + i/V]);
			beta.push_back(x);
		}
		else {
			beta.push_back(-100);
		}
		
		if (i < 20 and flag) { //x != x && 
			cout << "&&&& get LDA weights: n: " << n << " N: " << N 
				<< " V: " << V << " i: " << i
				<< " w[i]: " << weights[i] << " w[N+i/V]: " << weights[N + i/V]
				<< endl;
		}
	}
	return beta;
}

std::vector<double> Parameter::getGMMweights(int K, int D, int cnt, bool flag){

	std::vector<double> ww;
	/// sum_ll = weights[K*D*(D+1) + k]
	for (int k = 0; k < K; k++){
		for (int d = 0; d < D; d++){
			ww.push_back(weights[k*D +d] / weights[K*D*(D+1) + k]);
		}
	}
	for (int k = 0; k < K; k++){
		for (int d1 = 0; d1 < D; d1++){
			for (int d2 = 0; d2 < D; d2++){
				ww.push_back(weights[K*D + k*D*D + d1*D + d2] / weights[K*D*(D+1) + k]);
			}
		}
	}
	for (int k = 0; k < K; k++){
		ww.push_back(weights[K*D*(D+1) + k] / cnt);
	}
	return ww;
}