#include "GMM.h"
#include "util/Matrix.h"
#include "logging/logging.h"
#include <cmath>
#include <stdexcept>
using namespace std;

void GMM::init(const int xlength, const std::string & param)
{
	initBasic(xlength, param);
	K = stoi(param);
	D = xlength;
	VLOG(1) << "GMM param K:" << K << ", D:" << D;
}

std::string GMM::name() const{
	return "gmm";
}

bool GMM::dataNeedConstant() const{
	return false;
}

int GMM::lengthParameter() const
{
	return K*D + K*D*D + K;
}

/// given a data point and a parameter set
/// compute the best assignment
std::vector<double> GMM::predict(const std::vector<double>& x, 
	const std::vector<double>& w, const std::vector<double>& y) const 
{
	double likelihood = 0; // sum
	std::vector<double> g_dist;

	/// w[k] = w[K*D*(D+1) + k]
	/// mean[k][d] = w[k*D + d]
	double*** covariance = new double**[K];
	for (int i = 0; i < K; i++){
		covariance[i] = new double*[D];
		for (int j = 0; j < D; j++){
			covariance[i][j] = new double[D];
			for (int d = 0; d < D; d++){
				covariance[i][j][d] = w[K*D + i*D*D + j*D + d];
			}
		}
	}

	for (int k = 0; k < K; k++){
		double gd = Gaussian_Distribution(x, w, k, covariance[k]);
		g_dist.push_back(gd);
		likelihood += w[K*D*(D+1) + k] * gd;
	}

	g_dist.push_back(likelihood);

	delete[] covariance;
	return g_dist; /// assignment and dist
}

// double GMM::eudist(const std::vector<double>& x, const std::vector<double>& c, const int num) const{
// 	double dist = 0;
// 	for (int i = 0; i < x.size(); i++){
// 		double dis = (x[i] - c[i]/num);
// 		dist += dis * dis;
// 	}
// 	return dist;
// }

int GMM::classify(const double p) const
{
	return p >= 0.5 ? 1 : 0;
}

double GMM::loss(
	const std::vector<double>& pred, const std::vector<double>& label) const
{
	// double likelihood = pred.back();
	// cout << "likelihood for label " << label[0] << "; " << likelihood << endl;
	return pred.back();

//	if(std::isnan(cost) || std::isinf(cost)) // this std:: is needed for a know g++ bug
	// if(std::isinf(cost))
	// 	return MAX_LOSS;
	// else
	// 	return -cost;
}

std::vector<double> GMM::gradient(const std::vector<double>& x, const std::vector<double>& w, 
		const std::vector<double>& y, std::vector<double>* z) const
{
	std::vector<double> g_dist = predict(x, w);
	double sum = g_dist.back();
	// g_dist.pop_back();

	vector<double> ss;
	ss.assign(lengthParameter(), 0); // preallocate memory

	// double *ll = new double[K];
	for (int k = 0; k < K; k++){
		double ll = w[K*D*(D+1) + k] * g_dist[k] / sum;

		for (int d1 = 0; d1 < D; d1++){
			for (int d2 = 0; d2 < D; d2++){
				// new_covariance[j][k][l] += likelihood * 
				//      (data[i][k] - mean[j][k]) * (data[i][l] - mean[j][l]);
				ss[K*D + k*D*D + d1*D + d2] += 
						ll * (x[d1] - w[k*D + d1]) * (x[d2] - w[k*D + d2]);
			}

			// new_mean[j][k] += likelihood * data[i][k];
			ss[k*D + d1] += ll * x[d1];
		}
		ss[K*D + K*D*D + k] += ll;
	}
	// VLOG(3) << "new grad vector " << lengthParameter() << ", " << K << ", " << D
	// 	<< ", " << ss.size() << ", " << ss;

	if (ss.size() != z->size()){
		z = &ss;
		return ss;
	}
	else {
		std::vector<double> delta;
		for (int i = 0; i < ss.size(); i++){
			delta.push_back(ss[i] - (*z)[i]);
		}
		z = &ss;
		return delta;
	}
}

double GMM::Gaussian_Distribution(const std::vector<double>& data, 
			const std::vector<double>& mean, int k, double **covariance) const {
	double result;
	double sum = 0;

	double **inversed_covariance = new double*[D];

	Matrix matrix;

	for (int i = 0; i < D; i++){
		inversed_covariance[i] = new double[D];
	}
	matrix.Inverse("full", D, covariance, inversed_covariance);

	for (int i = 0; i < D; i++){
		double partial_sum = 0;

		for (int j = 0; j < D; j++){
			partial_sum += (data[j] - mean[k*D + j]) * inversed_covariance[j][i];
		}
		sum += partial_sum * (data[i] - mean[k*D + i]);
	}

	for (int i = 0; i < D; i++){
		delete[] inversed_covariance[i];
	}
	delete[] inversed_covariance;

	result = 1.0 / (pow(2 * 3.1415926535897931, D / 2.0) 
			* sqrt(matrix.Determinant("full", D, covariance))) * exp(-0.5 * sum);

	// if (_isnan(result) || !_finite(result)){
	// 	fprintf(stderr, "[Gaussian Distribution], [The covariance matrix is rank deficient], [result: %lf]\n", result);
	// }
	return result;
}
