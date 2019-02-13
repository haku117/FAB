#include "MLP.h"
#include "math/activation_func.h"
#include "util/Util.h"
#include <cmath>
#include <stdexcept>
using namespace std;

// -------- MLP --------

void MLP::init(const int xlength, const std::string & param)
{
	initBasic(xlength, param);
	nNodeLayer = getIntList(param, " ,-");
	if(nNodeLayer.empty() || xlength != nNodeLayer[0])
		throw invalid_argument("MLP parameter not valid or does not match dataset");
	// set n
	nLayer = static_cast<int>(nNodeLayer.size());
	proxy.init(nNodeLayer);
}

std::string MLP::name() const{
	return "mlp";
}

bool MLP::dataNeedConstant() const{
	return false;
}

int MLP::lengthParameter() const
{
	return proxy.lengthParameter();
}

std::vector<double> MLP::predict(
	const std::vector<double>& x, const std::vector<double>& w) const
{
	proxy.bind(&w);
	vector<double> mid = activateLayer(x, w, 0);
	for(int l = 1; l < nLayer - 1; ++l){
		mid = activateLayer(mid, w, l);
	}
	return mid;
}

int MLP::classify(const double p) const
{
	return p >= 0.5 ? 1 : 0;
}

constexpr double MAX_LOSS = 100;

double MLP::loss(const std::vector<double>& pred, const std::vector<double>& label) const {
	double res = 0.0;
	for(size_t i = 0; i < pred.size(); ++i){
		double t = pred[i] - label[i];
		res += t * t;
	}
	//return 0.5 * res;
	return res;
}

std::vector<double> MLP::gradient(
	const std::vector<double>& x, const std::vector<double>& w, const std::vector<double>& y) const
{
	proxy.bind(&w);
	// forward
	vector<vector<double>> buffer; // buffer for <pred>
	buffer.reserve(nLayer); // important: make sure earlier iterators valid all the time
	vector<const vector<double>*> output; // the output of each layer
	output.push_back(&x); // layer 0 same as x, the rest are <buffer>
	for(int l = 0; l < nLayer - 1; ++l){
		vector<double> mid = activateLayer(*output[l], w, l);
		buffer.push_back(move(mid));
		output.push_back(&buffer[l]);
	}
	// backward
	vector<double> grad(w.size());
	/*
	delta = np.array()
	for i in range(len(self.layers)-2, -1, -1):
		pred = self.output[i+1]
		if i == len(self.layers) - 2:
			error = pred - y
		else:
			error = np.dot(delta, self.w[i+1].T)
		delta = error * self.sigmoidPrime(pred)
		grad = np.dot(self.output[i].T, delta)
		self.w[i] += -self.lrate * grad
	*/
	vector<double> delta; // used for all but the first iteration (l==nLayer-2)
	for(int l = nLayer - 2; l >= 0; --l){
		// prepare
		const int n = nNodeLayer[l];
		const int m = nNodeLayer[l+1];
		const vector<double>& pred = *output[l+1];
		// error
		vector<double> error(m);
		if(l == nLayer - 2){
			//error = pred - y;
			for(int i = 0; i < m; ++i)
				error[i] = pred[i] - y[i];
		} else{
			//error = np.dot(delta, self.w[i + 1].T)
			MLPProxyLayer wl = proxy[l + 1];
			int h = nNodeLayer[l + 2];
			for(int i = 0; i < m; ++i){
				MLPProxyNode wn = wl[i];
				for(int j = 0; j < h; ++j)
					error[i] += delta[j] * wn[j]; // <delta> is left by last iteration
			}
		}
		// delta
		//delta = error * self.sigmoidPrime(pred)
		delta.resize(m);
		for(int i = 0; i < m; ++i){
			delta[i] = error[i] * sigmoid_derivative(pred[i]);
		}
		// grad
		//grad = np.dot(self.output[i].T, delta)
		MLPProxyLayer wl = proxy[l];
		for(int i = 0; i < n+1; ++i){
			MLPProxyNode wn = wl[i];
			double v = (i != n) ? (*output[l])[i] : 1.0;
			for(int j = 0; j < m; ++j){
				int offset = wn.position(j);
				grad[offset] = v * delta[j];
			}
		}
	}

	return grad;
}

double MLP::getWeight(const std::vector<double>& w, const int layer, const int from, const int to) const
{
	proxy.bind(&w);
	return proxy.get(layer, from, to);
}

std::vector<double> MLP::activateLayer(
	const std::vector<double>& x, const std::vector<double>& w, const int layer) const
{
	int n = nNodeLayer[layer];
	//assert(x.size() == n);
	int m = nNodeLayer[layer + 1];
	std::vector<double> res(m, 0.0); // # of real nodes in next layer
	MLPProxyLayer wl = proxy[layer]; // assume w has already been bound
	// real neuron part
	for(int i = 0; i < n; ++i){
		MLPProxyNode wn = wl[i];
		for(int j = 0; j < m; ++j){
			res[j] += x[i] * wn[j];
		}
	}
	// dummy neuron (constant value 1) part
	MLPProxyNode wn = wl[n];
	for(int j = 0; j < m; ++j){
		res[j] += wn[j];
		// activation function
		res[j] = sigmoid(res[j]);
	}
	//for(auto& v : res)
	//	v = sigmoid(v);
	return res;
}

