#include "Kmeans.h"
#include "mathfunc.h"
// #include "util/Util.h"
#include <cmath>
#include <stdexcept>
using namespace std;

void Kmeans::init(const int xlength, const std::string & param)
{
	initBasic(xlength, param);
	// std::vector<int> lst = getIntList(param, " ,-")[1];
	// k = lst[1];
	k = stoi(param);
	// int t = stoi(param);
	/// doesn't matter??
	// if(xlength != lst[0])
	// 	throw invalid_argument("Kmeans parameter does not match dataset");
}

std::string Kmeans::name() const{
	return "km";
}

bool Kmeans::dataNeedConstant() const{
	return false;
}

int Kmeans::lengthParameter() const
{
	return xlength + 1;
}

/// given a data point and a parameter set
/// compute the best assignments
///, const std::vector<double>& y
std::vector<double> Kmeans::predict(const std::vector<double>& x, 
	const std::vector<double>& w) const 
{
	vector<vector<double> > centroids;
	vector<int> counts;

	/// convert the parameter vector to centroids 
	for (int i = 0; i < k; i++) {
		move(w.begin() + i*k, w.begin() + i*k + xlength, back_inserter(centroids));
		counts.push_back(int(w[i*k, + xlength]));
	}

	/// compute new assignment
	double minDist = -1;
	int id = -1;
	for (int i = 0; i < centroids.size(); i++){
		double dist = eudist(x, centroids[i]);
		if (minDist == -1 || dist > minDist){
			minDist = dist;
			id = i;
		}
	}

	return { (double)id };

	// double t = w.back();
	// for (int i = 0; i < xlength; ++i) {
	// 	t += x[i] * w[i];
	// }
	// return { sigmoid(t) };
}

double Kmeans::eudist(const std::vector<double>& x, const std::vector<double>& c) const{
	double dist = 0;
	for (int i = 0; i < x.size(); i++){
		double dis = (x[i] - c[i]);
		dist += dis * dis;
	}
	return dist;
}

int Kmeans::classify(const double p) const
{
	return p >= 0.5 ? 1 : 0;
}

constexpr double MAX_LOSS = 100;

double Kmeans::loss(
	const std::vector<double>& pred, const std::vector<double>& label) const
{
	//double cost1 = label * log(pred);
	//double cost2 = (1 - label)*log(1 - pred);
	//return -(cost1 + cost2);
	// the above got overflow
	double cost;
	if(label[0] == 0.0){
		cost = log(1 - pred[0]);
	} else{
		cost = log(pred[0]);
	}
//	if(std::isnan(cost) || std::isinf(cost)) // this std:: is needed for a know g++ bug
	if(std::isinf(cost))
		return MAX_LOSS;
	else
		return -cost;
}

std::vector<double> Kmeans::gradient(
	const std::vector<double>& x, const std::vector<double>& w, const std::vector<double>& y) const
{
	std::vector<double> newY = predict(x, w);
	int id = (int) newY[0];

	vector<vector<double> > delta(k, vector<double>(xlength+1));
	/// compute delta change
	if (y[0] != id){
		for (int j = 0; j < xlength+1; j++){
			delta[id][j] += x[j];
			delta[y[0]][j] -= x[j];
		}
		delta[id][xlength]++;
		delta[y[0]][xlength]--;
	}

	vector<double> grad;
	grad.reserve( k * w.size()); // preallocate memory
	for (int i = 0; i < k; i++){
		grad.insert( grad.end(), delta[i].begin(), delta[i].end() );
	}

	/// update assignment
	// y = move(newY);

	return grad;
}

