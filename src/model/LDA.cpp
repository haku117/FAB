#include "LDA.h"
#include "logging/logging.h"
#include "util/Util.h"
#include <cmath>
#include <stdexcept>
#include <cassert>
using namespace std;

void LDA::init(const int xlength, const std::string & param)
{
	// std::vector<int> tokens = parseParam(param);
	// nnx = tokens[1];
	// nny = tokens[2];
	num_topics = stoi(param);
	// num_terms = xlength;
	num_terms = 10434;
	alpha = 0.2;
	initBasic(xlength, param); // ??

	// intail
    // for (int k = 0; k < num_topics; k++){
    // 	for (int n = 0; n < num_terms; n++){
	// 		oldSS.push_back(0);
	// 	}
	// 	oldSS.push_back(0);
	// }

	// VLOG(3) << "Init LDA: " << num_topics << ", " << num_terms << ", " 
	// 		<< xlength << ", " << oldSS.size();
	// for (int i = 0; i < num_topics * (num_terms + 1); i++){
	// 	beta.push_back(0);
	// }

	// initial beta

	// if(xlength != lst[0])
	// 	throw invalid_argument("LDA parameter does not match dataset");
}

std::string LDA::name() const{
	return "lda";
}

bool LDA::dataNeedConstant() const{
	return false;
}

int LDA::lengthParameter() const
{
	return num_topics * (num_terms + 1); /// ss_word
}

/// E-step:
/// given a doc and a parameter set
/// compute the liklihood
std::vector<double> LDA::predict(const std::vector<double>& x, 
	const std::vector<double>& w, const std::vector<double>& y) const 
{
	size_t docN = x.size();
	int k, n, totalW = 0;
	for (auto& ye : y)
    	totalW += ye;
	
	// VLOG(3) << "x: " << x << ",y: " << y;

	// initialize gamma and phi
	vector<double> gamma, oldphi, digamma_gam;
	double initGamma = alpha + totalW/num_topics;
	double initDigamma = digamma(initGamma);
	for (k = 0; k < num_topics; k++){
		gamma.push_back(initGamma);
		digamma_gam.push_back(initDigamma);
		oldphi.push_back(0);
	}
	// VLOG(3) << "gamma: " << gamma;
	// VLOG(3) << "beta: " << w;
	// VLOG(3) << "digamma_gam: " << digamma_gam;
	vector<double> phi;
	for (n = 0; n < docN; n++){
		for (int k = 0; k < num_topics; k++){
			phi.push_back(1.0/num_topics);
		}
	}
	// VLOG(3) << "docN:" << docN << "  phi size:" << phi.size();
	// VLOG(3) << "phi: " << phi;
	
	double phisum = 0;
	// train gamma and phi
	for (n = 0; n < docN; n++){
		phisum = 0;
		for (k = 0; k < num_topics; k++){
			// VLOG_IF(n * num_topics + k >= phi.size(), 3) << "??" << n * num_topics + k 
			// 	<< "," << n << "," << k;
			// VLOG_IF(k * num_terms + x[n] >= beta.size(), 3) << "??" << k * num_terms + x[n] 
			// 	<< "," << k << "," << x[n];

			oldphi[k] = phi[n * num_topics + k];
			phi[n * num_topics + k] = digamma_gam[k] + w[k * num_terms + x[n]];

			// cout << w[k * num_terms + x[n]] << "," << phi[n * num_topics + k] << ";";
				// model->log_prob_w[k][doc->words[n]];
			if (k > 0)
				phisum = log_sum(phisum, phi[n * num_topics + k]);
			else
				phisum = phi[n * num_topics + k]; // note, phi is in log space
		}

		for (k = 0; k < num_topics; k++){
			phi[n * num_topics + k] = exp(phi[n * num_topics + k] - phisum);
			gamma[k] += y[n]*(phi[n * num_topics + k] - oldphi[k]);
			// !!! a lot of extra digamma's here because of how we're computing it
			// !!! but its more automatically updated too.
			digamma_gam[k] = digamma(gamma[k]);
		}
	}

	// VLOG(3) << "phiT: " << phi;
	// VLOG(3) << "gamma: " << gamma;

	// compute liklihood
	vector<double> dig;
	double gamma_sum = 0;
	for (k = 0; k < num_topics; k++){
		double dg = digamma(gamma[k]);
		dig.push_back(dg);
		gamma_sum += gamma[k];
	}
	double dig_sum = digamma(gamma_sum);

	double likelihood = lgamma(alpha * num_topics)
					- num_topics * lgamma(alpha) - (lgamma(gamma_sum));

	// if(likelihood != likelihood) {
	// 	cout << "init ll NaN @@@ gamma_sum : " << gamma_sum 
	// 	 	<< " lgamma: " << lgamma(gamma_sum) << endl;
	// 	VLOG(2) << "beta: " << w;

		// vector<double> gamma, oldphi, digamma_gam;
		// double initGamma = alpha + totalW/num_topics;
		// double initDigamma = digamma(initGamma);
		// for (k = 0; k < num_topics; k++){
		// 	gamma.push_back(initGamma);
		// 	digamma_gam.push_back(initDigamma);
		// 	oldphi.push_back(0);
		// }
		// VLOG(2) << "gamma: " << gamma;
		// VLOG(2) << "digamma_gam: " << digamma_gam;
		// vector<double> phi;
		// for (n = 0; n < docN; n++){
		// 	for (int k = 0; k < num_topics; k++){
		// 		phi.push_back(1.0/num_topics);
		// 	}
		// }
		// VLOG(2) << "docN:" << docN << "  phi size:" << phi.size();
		// // VLOG(2) << "phi: " << phi;
		// // train gamma and phi
		// for (n = 0; n < docN; n++){
		// 	phisum = 0;
		// 	for (k = 0; k < num_topics; k++){
		// 		oldphi[k] = phi[n * num_topics + k];
		// 		phi[n * num_topics + k] = digamma_gam[k] + w[k * num_terms + x[n]];
		// 		if (k > 0)
		// 			phisum = log_sum(phisum, phi[n * num_topics + k]);
		// 		else
		// 			phisum = phi[n * num_topics + k];
		// 	}

		// 	for (k = 0; k < num_topics; k++){
		// 		phi[n * num_topics + k] = exp(phi[n * num_topics + k] - phisum);
		// 		gamma[k] += y[n]*(phi[n * num_topics + k] - oldphi[k]);
		// 		digamma_gam[k] = digamma(gamma[k]);
		// 	}
		// }
		// VLOG(2) << "phiT: " << phi;
		// VLOG(2) << "gamma: " << gamma;

	// 	phi.push_back(likelihood);
	// 	return phi;
	// }

    for (k = 0; k < num_topics; k++){
		likelihood += (alpha - 1)*(dig[k] - dig_sum) 
			+ lgamma(gamma[k]) - (gamma[k] - 1)*(dig[k] - dig_sum);

		// if(likelihood != likelihood)
		// 	cout << " ** ll1: " << likelihood << ", k: " << k << ", gamma[k]: " << gamma[k]
		// 		<< ", lgamma: " << lgamma(gamma[k]) << ", digk: " << dig[k] 
		// 		<< ", dig_sum: " << dig_sum << ", gamma_sum: " << gamma_sum << endl;

		for (n = 0; n < docN; n++){
            if (phi[n * num_topics + k] > 0){
                likelihood += y[n] * phi[n * num_topics + k] * ( dig[k] - dig_sum
					- log(phi[n * num_topics + k]) + w[k * num_terms + x[n]]);
            }
        }
		// if(likelihood != likelihood)
		// 	cout << " ** ll2: " << likelihood;
    }

	// cout << " ## llTT: " << likelihood << endl;
	// VLOG_IF(docN > 300, 2) << "   likelihood for " << docN << " : " << likelihood;
	phi.push_back(likelihood);
	return phi;
}

int LDA::classify(const double p) const
{
	return p >= 0.5 ? 1 : 0;
}

constexpr double MAX_LOSS = 100;

double LDA::loss(
	const std::vector<double>& pred, const std::vector<double>& label) const
{
	double likelihood = pred[pred.size()-1];
	// cout << "likelihood for label " << label[0] << "; " << likelihood << endl;
	return likelihood;
}

std::vector<double> LDA::gradient(const std::vector<double>& x, const std::vector<double>& w, 
		const std::vector<double>& y, std::vector<double>* z) const
{
	std::vector<double> phi = predict(x, w, y);
	phi.pop_back(); // remove likelihood
	int k, n;
	size_t docN = x.size();
	// int ssSize = oldSS.size();

	// update??
	vector<double> CW, CTT;
    for (k = 0; k < num_topics; k++){
    	for (n = 0; n < num_terms + 1; n++){
			CW.push_back(0);
		}
		// CTT.push_back(0);
	}
	int ssSize = CW.size();
    for (n = 0; n < docN; n++){
        for (k = 0; k < num_topics; k++){
			double accu = y[n] * phi[n * num_topics + k];
            CW[k * num_terms + x[n]] += accu;
			CW[ssSize - num_topics + k] += accu;
            // CTT[k] += accu;
        }
    }
    // CW.insert(CW.end(), CTT.begin(), CTT.end());
	// VLOG_IF(CW[0] != CW[0], 3) << "gradient: " << docN << ", " << CW.size() << ", " << CW;
	// VLOG_IF(CW[0] != CW[0], 3) << "phi: " << phi;
	if (CW.size() != z->size()){
		z = &CW;
		return CW;
	}
	else {
		std::vector<double> delta;
		for (int i = 0; i < CW.size(); i++){
			delta.push_back(CW[i] - (*z)[i]);
		}
		z = &CW;
		return delta;
	}
}

// void LDA::updateLocalZ(std::vector<double>& zz){
// 	oldSS = move(zz);
// }

// std::vector<double> LDA::computeDelta(std::vector<double>& zz){
// 	VLOG(3) << "worker compute delta: " << zz.size() << ", " << zz;
// 	VLOG(3) << "oldSS: " << oldSS.size() << ", " << oldSS;
// 	if (oldSS.size() != zz.size()) {
// 		oldSS = zz;
// 		return move(zz);
// 	}
// 	std::vector<double> delta;
// 	for (int i = 0; i < zz.size(); i++) {
// 		delta.push_back(zz[i] - oldSS[i]);
// 		oldSS[i] = zz[i];
// 	}
// 	return delta;
// }