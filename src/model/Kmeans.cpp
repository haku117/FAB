#include "Kmeans.h"
#include "logging/logging.h"
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
	return (xlength + 1) * k;
}

/// given a data point and a parameter set
/// compute the best assignment
std::vector<double> Kmeans::predict(const std::vector<double>& x, 
	const std::vector<double>& w, const std::vector<double>& y) const 
{
	vector<vector<double> > centroids;
	vector<int> counts;

	/// convert the parameter vector to centroids 
	for (int i = 0; i < k; i++) {
		vector<double> cc(w.begin() + i*(xlength+1), w.begin() + i*(xlength+1) + xlength);
		centroids.push_back(cc);
		counts.push_back(int(w[i*(xlength+1) + xlength]));
	}

	/// compute new assignment
	double minDist = -1;
	int id = -1;
	for (int i = 0; i < k; i++){
		double dist = eudist(x, centroids[i], counts[i]);
		if (minDist == -1 || dist < minDist){
			minDist = dist;
			id = i;
		}
	}

	return { (double)id, minDist }; /// assignment and dist
}

double Kmeans::eudist(const std::vector<double>& x, const std::vector<double>& c, const int num) const{
	double dist = 0;
	for (int i = 0; i < x.size(); i++){
		double dis = (x[i] - c[i]/num);
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
	if (pred.size() != 2){
		VLOG(4) << "Incorrect pred loss for Kmeans " << pred;
		return MAX_LOSS;
	}
	else  
		return pred[1];

//	if(std::isnan(cost) || std::isinf(cost)) // this std:: is needed for a know g++ bug
	// if(std::isinf(cost))
	// 	return MAX_LOSS;
	// else
	// 	return -cost;
}

std::vector<double> Kmeans::gradient(const std::vector<double>& x, const std::vector<double>& w, 
		const std::vector<double>& y, std::vector<double>* z) const
{
	std::vector<double> newZ = predict(x, w);
	int id = (int) newZ[0];
	int oldID = (*z)[0];
	// double oldDist = (*z)[1];

	VLOG(5) << "KM grad newZ " << id << " old Z " << oldID << " xlen " << xlength;

	vector<vector<double> > delta(k, vector<double>(xlength+1, 0));
	// VLOG(5) << "init delta " << delta.size() << "\t" << x;
	/// compute delta change
	if (oldID != id){
		for (int j = 0; j < xlength; j++){
			delta[id][j] += x[j];
		}
		delta[id][xlength] += 1;
		if(oldID >= 0){
			for (int j = 0; j < xlength; j++){
				delta[oldID][j] -= x[j];
			}
			delta[oldID][xlength] -= 1;
		}
	}

	VLOG(5) << "new delta " << delta;

	vector<double> grad;
	grad.reserve(w.size()); // preallocate memory
	for (int i = 0; i < k; i++){
		grad.insert( grad.end(), delta[i].begin(), delta[i].end() );
	}

	VLOG(5) << "new grad vector " << grad;
			/// update assignment
	(*z)[0] = id;
	if((*z)[1] < 0.00001)
		grad.push_back(newZ[1]); //// improvement
	else
		grad.push_back((*z)[1] - newZ[1]); //// improvement
	if (abs(newZ[1]) > 1000){
		VLOG(1) << "??????????????????? DIST: " << newZ[1] << "; " << (*z)[1];
	}
	grad.push_back(newZ[1]); //// improvement
	(*z)[1] = newZ[1];
	return grad;
}
