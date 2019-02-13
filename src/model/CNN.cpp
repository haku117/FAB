#include "CNN.h"
#include "util/Util.h"
#include <cmath>
#include <stdexcept>
using namespace std;

// -------- CNN --------

void CNN::init(const int xlength, const std::string & param)
{
	initBasic(xlength, param);
    // example: 10*10-4,c,3*3-1,a,relu-1,p,max,2*2-1,f
	// example: 10-4,c,3-1,a,relu-1,p,max,2-1,f
    // format: <n>,<type>[,<shape>]
    //     shape of convolutional node: <k1>*<k2>
    //     shape of fully-connected node: none
    try{
		proxy.init(param);
    }catch(exception& e){
		throw invalid_argument(e.what());
    }
	// check input layer size
	int n = 1;
	for(auto& v : proxy.shapeLayer[0])
		n *= v;
	if(xlength != n)
		throw invalid_argument("The dataset does not match the input layer of the network");
	// check FC layer
	for(size_t i = 0; i < proxy.nLayer; ++i){
		if(i != proxy.nLayer - 1 && proxy.typeLayer[i] == LayerType::FC){
			throw invalid_argument("Only the last layer can be a FC layer.");
		} else if(i == proxy.nLayer - 1 && proxy.typeLayer[i] != LayerType::FC){
			throw invalid_argument("The last layer must be a FC layer.");
		}
	}
	// set local parameters
	nLayer = proxy.nLayer;
	nNodeLayer = proxy.nNodeLayer;
	nWeight = proxy.lengthParameter();
}

std::string CNN::name() const{
	return "cnn";
}

bool CNN::dataNeedConstant() const{
	return false;
}

int CNN::lengthParameter() const
{
	return nWeight;
}

std::vector<double> CNN::predict(
	const std::vector<double>& x, const std::vector<double>& w) const
{
	vector<vector<double>> input;
	vector<vector<double>> output;
	input.push_back(x);
	// apart from the last FC layer, all nodes work on a single feature
	for(int i = 1; i < proxy.nLayer - 1; ++i){
		output.clear();
		// apply one node on each previous features repeatedly
		//assert(proxy.nFeatureLayer[i - 1] * proxy.nNodeLayer[i] == proxy.nFeatureLayer[i]);
		for(int j = 0; j < proxy.nNodeLayer[i]; ++j){
			for(int k = 0; k < proxy.nFeatureLayer[i - 1]; ++k){
				output.push_back(proxy.nodes[i][j]->predict(input[k], w));
			}
		}
		input = move(output);
	}
	// the last FC layer
	vector<double> res;
	int i = proxy.nLayer - 1;
	for(int j = 0; j < proxy.nNodeLayer[i]; ++j){
		FCNode1D* p = dynamic_cast<FCNode1D*>(proxy.nodes[i][j]);
		res.push_back(p->predict(input, w));
	}
	return res;
}

int CNN::classify(const double p) const
{
	return p >= 0.5 ? 1 : 0;
}

constexpr double MAX_LOSS = 100;

double CNN::loss(const std::vector<double>& pred, const std::vector<double>& label) const {
	double res = 0.0;
	for(size_t i = 0; i < pred.size(); ++i){
		double t = pred[i] - label[i];
		res += t * t;
	}
	//return 0.5 * res;
	return res;
}

std::vector<double> CNN::gradient(
	const std::vector<double>& x, const std::vector<double>& w, const std::vector<double>& y) const
{
	// forward
	vector<vector<vector<double>>> mid; // layer -> feature -> value
	mid.reserve(proxy.nLayer); // intermediate result of all layers
	mid.push_back({ x });
	for(int i = 1; i < proxy.nLayer - 1; ++i){ // apart from the input and output (FC) layers
		const vector<vector<double>>& input = mid[i - 1];
		vector<vector<double>> output;
		// apply one node on each previous features repeatedly
		//assert(proxy.nFeatureLayer[i - 1] * proxy.nNodeLayer[i] == proxy.nFeatureLayer[i]);
		for(int j = 0; j < proxy.nNodeLayer[i]; ++j){
			for(int k = 0; k < proxy.nFeatureLayer[i - 1]; ++k){
				output.push_back(proxy.nodes[i][j]->predict(input[k], w));
			}
		}
		mid.push_back(move(output));
	}
	vector<vector<double>> output;
	vector<FCNode1D*> finalNodes;
	for(int j = 0; j < proxy.nNodeLayer.back(); ++j){
		FCNode1D* p = dynamic_cast<FCNode1D*>(proxy.nodes.back()[j]);
		finalNodes.push_back(p);
		output.push_back({ p->predict(mid[proxy.nLayer - 2], w) });
	}
	mid.push_back(output);
	// back propagate
	// BP (last layer)
	vector<double> grad(w.size());
	vector<vector<double>> partial; // partial gradient
	//assert(y.size() == mid.back().size());
	for(size_t i = 0; i < y.size(); ++i){ // last FC layer
		double output = mid.back()[i][0];
		double pg = output - y[i];
		FCNode1D* p = finalNodes[i];
		vector<vector<double>> temp = p->gradient(grad, mid[proxy.nLayer - 2], w, output, pg);
		if(i == 0){
			partial = move(temp);
		} else{
			for(size_t a = 0; a < temp.size(); ++a)
				for(size_t b = 0; b < temp[a].size(); ++b)
					partial[a][b] += temp[a][b];
		}
	}
	// BP (0 to n-1 layer)
	for(int i = proxy.nLayer - 2; i > 0; --i){ // layer
		int oidx = 0;
		vector<vector<double>> newPartialGradient(proxy.nFeatureLayer[i]);
		for(int j = 0; j < proxy.nNodeLayer[i]; ++j){ // node
			for(int k = 0; k < proxy.nFeatureLayer[i - 1]; ++k){ // feature (input)
				const vector<double>& input = mid[i - 1][k];
				const vector<double>& output = mid[i][oidx];
				const vector<double>& pg = partial[oidx];
				++oidx;
				NodeBase* p = proxy.nodes[i][j];
				vector<double> npg = p->gradient(grad, input, w, output, pg);
				if(j == 0){
					newPartialGradient[k] = move(npg);
				} else{
					for(size_t a = 0; a < npg.size(); ++a)
						newPartialGradient[k][a] += npg[a];
				}
			}
		}
		partial = move(newPartialGradient);
	}
	return grad;
}
