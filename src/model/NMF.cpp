#include "NMF.h"
#include "logging/logging.h"
#include "util/Util.h"
#include <cmath>
#include <stdexcept>
using namespace std;

void NMF::init(const int xlength, const std::string & param)
{
	lambda = 0.01; //// default lambda
	size_t pos = 0;
	std::vector<int> tokens = parseParam(param);
	nnx = tokens[1];
	nny = tokens[2];
	initBasic(tokens[0], param);
}

std::string NMF::name() const{
	return "nmf";
}

bool NMF::dataNeedConstant() const{
	return false;
}

int NMF::lengthParameter() const
{
	return (nnx + nny) * xlength;
}

/// given the position of the dp and the parameters
/// compute the predicted value at that position 
/// may not need to use??
std::vector<double> NMF::predict(const std::vector<double>& x, 
	const std::vector<double>& w, const std::vector<double>& y) const 
{
	int xi = int(x[0]);
	int yi = int(x[1]);
	vector<double>::const_iterator first = w.begin() + (xlength * xi);
	vector<double>::const_iterator last = w.begin() + (xlength * (xi+1));
	vector<double> wu(first, last);
	first = w.begin() + (xlength * (nnx+yi));
	last = w.begin() + (xlength * (nnx+yi+1));
	vector<double> hi(first, last);

	double t = 0;
	for (int i = 0; i < xlength; ++i) {
		t += wu[i] * hi[i];
	}
	return { t };
}

// double NMF::eudist(const std::vector<double>& x, const std::vector<double>& c, const int num) const{
// 	double dist = 0;
// 	for (int i = 0; i < x.size(); i++){
// 		double dis = (x[i] - c[i]/num);
// 		dist += dis * dis;
// 	}
// 	return dist;
// }

int NMF::classify(const double p) const
{
	return p >= 0.5 ? 1 : 0;
}

double NMF::loss(
	const std::vector<double>& pred, const std::vector<double>& label) const
{
	double err = label[0] - pred[0];
	return err*err;
}

std::vector<double> NMF::gradient(const std::vector<double>& x, const std::vector<double>& w, 
		const std::vector<double>& y, std::vector<int>* z) const
{
	// double err = y[0] - predict(x, w)[0];
	int xi = int(x[0]);
	int yi = int(x[1]);

	vector<double>::const_iterator first = w.begin() + (xlength * xi);
	vector<double>::const_iterator last = w.begin() + (xlength * (xi+1));
	vector<double> wu(first, last);
	first = w.begin() + (xlength * (nnx+yi));
	last = w.begin() + (xlength * (nnx+yi+1));
	vector<double> hi(first, last);

	double pred = 0;
	for (int i = 0; i < xlength; ++i) {
		pred += wu[i] * hi[i];
	}
	double err = y[0] - pred;

	////// efficient delta
	// vector<double> deltaWu(xlength, 0);
	// vector<double> deltaHi(xlength, 0);
	// std::vector<double> delta;
	// delta.push_back(xi);
	// for (int j = 0; j < xlength; j++){
	// 	// deltaWu[j] = err * hi[j] - lambda * wu[j];
	// 	// deltaHi[j] = err * wu[j] - lambda * hi[j];
	// 	delta.push_back(err * hi[j] - lambda * wu[j]);
	// }
	// delta.push_back(nnx + yi);
	// for (int j = 0; j < xlength; j++){
	// 	delta.push_back(err * wu[j] - lambda * hi[j]);
	// }

	////// normal delta
	std::vector<double> delta(w.size(), 0);
	for (int j = 0; j < xlength; j++){
		delta[xi * xlength + j] = err * hi[j] - lambda * wu[j];
		delta[(nnx + yi) * xlength + j] = err * wu[j] - lambda * hi[j];
	}

	// VLOG(3) << "x: " << x << ",y: " << y << "\tw size:" << w.size() 
	// 	<< "\td size:" << delta.size() << "\terr:" << err;
	
	return delta;
}
