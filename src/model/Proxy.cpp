#include "Proxy.h"
#include "util/Util.h"
#include "mathfunc.h"
#include <algorithm>
#include <numeric>
#include <regex>
#include <cassert>

using namespace std;

void Proxy::init(const std::string& param){
    vector<string> strLayer = getStringList(param, "-");
    nLayer = strLayer.size();
    // raw string format: R"(...)"
    string srShapeNode = R"((\d+(?:\*\d+)*))"; // v1[*v2[*v3[*v4]]]
    //regex ri(srShapeNode); // input layer
    regex ra(R"((\d+),a,(sigmoid|relu))"); // activation layer, i.e.: 4,a,relu
    regex rc(R"((\d+),c,)"+srShapeNode); // convolutional layer, i.e.: 4,c,4*4
    regex rp(R"((\d+),p,(max|mean|min),)"+srShapeNode); // pooling layer, i.e.: 1,p,max,3*3
	regex rf(R"((\d+),f)"); // fully-connected layer, i.e.: 4,f

    typeLayer.resize(nLayer);
    shapeNode.resize(nLayer);
    unitNode.resize(nLayer);
    shapeLayer.resize(nLayer);
    ndimLayer.resize(nLayer);
	nWeightNode.resize(nLayer);
	weightOffsetLayer.resize(nLayer + 1);
	nodes.resize(nLayer);

	// input (first layer)
	nNodeLayer[0] = 1;
    typeLayer[0] = LayerType::Input;
	unitNode[0] = { 1 };
	shapeNode[0] = shapeLayer[0] = getShape(strLayer[0]);
    ndimLayer[0] = shapeLayer[0].size();
    for(size_t i = 1; i<strLayer.size(); ++i){
        smatch m;
        if(regex_match(strLayer[i], m, ra)){ // activation
            nNodeLayer[i] = stoi(m[1]);
            if(m[2] == "sigmoid")
                typeLayer[i] = LayerType::ActSigmoid;
            else if(m[2] == "relu")
                typeLayer[i] = LayerType::ActRelu;
            else if(m[2] == "tanh")
                typeLayer[i] = LayerType::ActTanh;
			unitNode[i] = { 1 };
			shapeNode[i] = shapeNode[i - 1];
			setLayerParameter(i);
			generateNode(i);
        }else if(regex_match(strLayer[i], m, rc)){ // convolutional
            nNodeLayer[i] = stoi(m[1]);
            typeLayer[i] = LayerType::Conv;
            unitNode[i] = getShape(m[2]);
			size_t p = shapeLayer[i - 1].size() - unitNode[i].size();
			assert(p == 0 || p == 1);
			for(size_t j = 0; j < unitNode[i].size(); ++j){
				shapeNode[i].push_back(shapeLayer[i - 1][p + j] - unitNode[i][j] + 1);
			}
			setLayerParameter(i);
			generateNode(i);
        }else if(regex_match(strLayer[i], m, rp)){ // pool
            nNodeLayer[i] = stoi(m[1]);
			string type = m[2];
			if(type == "max")
				typeLayer[i] = LayerType::PoolMax;
			else if(m[2] == "mean")
				typeLayer[i] = LayerType::PoolMean;
			else if(m[2] == "min")
				typeLayer[i] = LayerType::PoolMin;
            unitNode[i] = getShape(m[3]);
			size_t p = shapeLayer[i - 1].size() - unitNode[i].size();
			assert(p == 0 || p == 1);
			for(size_t j = 0; j < unitNode[i].size(); ++j){
				int v = (shapeLayer[i - 1][p + j] + unitNode[i][j] - 1) / unitNode[i][j];
				shapeNode[i].push_back(v);
			}
			setLayerParameter(i);
			generateNode(i);
		}else if(regex_match(strLayer[i], m, rf)){ // fully-connected
			nNodeLayer[i] = stoi(m[1]);
			typeLayer[i] = LayerType::FC;
			unitNode[i] = shapeLayer[i - 1];
			shapeNode[i] = { 1 };
			shapeLayer[i] = { nNodeLayer[i], 1 };
			ndimLayer[i] = 2;
			generateNode(i);
		}
    }
}
  
int Proxy::lengthParameter() const
{
	return weightOffsetLayer[nLayer];
}

std::vector<int> Proxy::getShape(const string& str){
    return getIntList(str, "*");
}

int Proxy::getSize(const std::vector<int>& ShapeNode){
    if(ShapeNode.empty())
        return 0;
    int r = 1;
    for(auto& v : ShapeNode)
        r*=v;
    return r;
}

void Proxy::setLayerParameter(int i){
	if(shapeLayer[i - 1].size() == shapeNode[i].size()){
		nFeatureLayer[i] = nNodeLayer[i];
	}else if(shapeLayer[i - 1].size() == shapeNode[i].size() + 1){
		nFeatureLayer[i] = shapeLayer[i - 1][0] * nNodeLayer[i];
	} else{
		throw invalid_argument("shape between layer "
			+ to_string(i - 1) + " and " + to_string(i) + " does not match.");
	}
	shapeLayer[i].push_back(nFeatureLayer[i]);
	for(size_t j = 0; j < shapeNode[i].size(); ++j)
		shapeLayer[i].push_back(shapeNode[i][j]);
	ndimLayer[i] = shapeLayer[i].size();
}

void Proxy::generateNode(const int i)
{
	int offset = weightOffsetLayer[i];
	vector<NodeBase*>& vec = nodes[i];
	for(int j = 0; j < nNodeLayer[i]; ++j){
		NodeBase* p = nullptr;
		switch(typeLayer[i])
		{
		case LayerType::ActRelu:
			p = new ReluNode(offset, unitNode[i]);
			break;
		case LayerType::ActSigmoid:
			p = new SigmoidNode(offset, unitNode[i]);
			break;
		case LayerType::Conv:
			p = new ConvNode1D(offset, unitNode[i]);
			break;
		case LayerType::PoolMax:
			p = new PoolMaxNode1D(offset, unitNode[i]);
			break;
		case LayerType::FC:
			p = new FCNode1D(offset, unitNode[i]);
			break;
		default:
			throw invalid_argument("try to generate an unsupported node type.");
			break;
		}
		offset += p->nweight();
		nWeightNode[i] = p->nweight();
		vec.push_back(p);
	}
	weightOffsetLayer[i + 1] = offset;
}

// nodes

NodeBase::NodeBase(const size_t offset, const std::vector<int>& shape)
	: off(offset), shape(shape)
{
	nw = accumulate(shape.begin(), shape.end(), 1,
		[](double a, double b){return a * b; }
	);
}

size_t NodeBase::nweight() const
{
	return nw;
}

InputNode::InputNode(const size_t offset, const std::vector<int>& shape)
	: NodeBase(offset, shape)
{}

std::vector<double> InputNode::predict(const std::vector<double>& x, const std::vector<double>& w)
{
	return x;
}

std::vector<double> InputNode::gradient(std::vector<double>& grad, const std::vector<double>& x,
	const std::vector<double>& w, const std::vector<double>& y, const std::vector<double>& pre)
{
	return std::vector<double>();
}

// 1d convolutional node

ConvNode1D::ConvNode1D(const size_t offset, const std::vector<int>& shape)
	: NodeBase(offset, shape), k(shape[0])
{
	assert(shape.size() == 1);
	assert(k > 0);
}

std::vector<double> ConvNode1D::predict(const std::vector<double>& x, const std::vector<double>& w)
{
	const size_t ny = x.size() - k + 1;
	std::vector<double> res(ny);
    for(size_t i=0; i<ny; ++i){
        double t = 0.0;
        for(size_t j=0; j<k; ++j)
            t += x[i+j] * w[off+j];
        res[i] = t;
    }
    return res;
}

std::vector<double> ConvNode1D::gradient(std::vector<double>& grad, const std::vector<double>& x,
	const std::vector<double>& w, const std::vector<double>& y, const std::vector<double>& pre)
{
	assert(x.size() == y.size() + nw - 1);
	assert(y.size() == pre.size());
	const size_t nx = x.size();
	const size_t ny = y.size();
	// dy/dw
	for(size_t i = 0; i < nw; ++i){
		double t = 0.0;
		for(size_t j = 0; j < ny; ++j){
			t += pre[j] * x[j + i];
		}
		grad[off + i] = t;
	}
	// dy/dx
	std::vector<double> res(nx);
	for(size_t i = 0; i < nx; ++i){
		double t = 0.0;
		// cut the first and the last
		for(size_t j = (i < ny ? 0 : i - ny + 1); j < nw && i >= j; ++j){
			t += pre[i - j] * w[off + j];
		}
		res[i] = t;
	}
	return res;
}

// relu node

ReluNode::ReluNode(const size_t offset, const std::vector<int>& shape)
	: NodeBase(offset, shape)
{
	nw = 1;
}

std::vector<double> ReluNode::predict(const std::vector<double>& x, const std::vector<double>& w)
{
    const size_t n = x.size();
    std::vector<double> res(n);
    double t = 0.0;
    for(size_t i=0; i<n; ++i){
        res[i] = relu(x[i] + w[off]);
    }
    return res;
}

std::vector<double> ReluNode::gradient(std::vector<double>& grad, const std::vector<double>& x,
	const std::vector<double>& w, const std::vector<double>& y, const std::vector<double>& pre)
{
	assert(x.size() == y.size());
	const size_t n = y.size();
	std::vector<double> res(n); // dy/dx
	double s = 0.0;
	for(size_t i = 0; i < n; ++i){
		double d = relu_derivative(x[i] + w[off]);
		s += pre[i] * d * x[i]; // dy/dw
		res[i] = pre[i] * d * w[off]; // dy/dx
	}
	grad[off] = s / n;
	return res;
}

// Sigmoid node

SigmoidNode::SigmoidNode(const size_t offset, const std::vector<int>& shape)
	: NodeBase(offset, shape)
{
	nw = 1;
}

std::vector<double> SigmoidNode::predict(const std::vector<double>& x, const std::vector<double>& w)
{
    const size_t n = x.size();
    std::vector<double> res(n);
    for(size_t i=0; i<n; ++i){
        res[i] = sigmoid(x[i] + w[off]);
    }
    return res;
}

std::vector<double> SigmoidNode::gradient(std::vector<double>& grad, const std::vector<double>& x,
	const std::vector<double>& w, const std::vector<double>& y, const std::vector<double>& pre)
{
	assert(x.size() == y.size());
    const size_t n = y.size();
	std::vector<double> res(n);
    double s = 0.0;
    for(size_t i=0; i<n; ++i){
		//double d = sigmoid_derivative(x[i] + w[off], y[i]);
		double d = sigmoid_derivative(0.0, y[i]);
		s += pre[i] * d * x[i]; // dy/dw
		res[i] = pre[i] * d * w[off]; // dy/dx
    }
    grad[off] = s / n;
	return res;
}

// 1d pooling - max node

PoolMaxNode1D::PoolMaxNode1D(const size_t offset, const std::vector<int>& shape)
	: NodeBase(offset, shape)
{
	nw = 0;
	assert(shape.size() == 1);
}

std::vector<double> PoolMaxNode1D::predict(const std::vector<double>& x, const std::vector<double>& w)
{
	const size_t step = shape[0];
	const size_t n = (x.size() + step - 1) / step;
	vector<double> res(n);
	for(size_t i = 0; i < n; ++i){
		double v = x[i*step];
		size_t limit = min((i + 1)*step, x.size());
		for(size_t j = i*step + 1; j < limit; ++j)
			v = max(v, x[j]);
		// TODO: store the max index for gradient
		res.push_back(v);
	}
	return res;
}

std::vector<double> PoolMaxNode1D::gradient(std::vector<double>& grad, const std::vector<double>& x,
	const std::vector<double>& w, const std::vector<double>& y, const std::vector<double>& pre)
{
	const size_t step = shape[0];
	const size_t ny = y.size();
	vector<double> res(x.size(), 0.0);	
	for(size_t i = 0; i < ny; ++i){
		size_t limit = min((i + 1)*step, x.size());
		for(size_t j = i * step; j < limit; ++j){
			if(x[j] == y[i])
				res[j] = 1.0;
		}
	}
	return res;
}

// FC 1D node

FCNode1D::FCNode1D(const size_t offset, const std::vector<int>& shape)
	: NodeBase(offset, shape)
{}

double FCNode1D::predict(const std::vector<std::vector<double>>& x, const std::vector<double>& w)
{
	const size_t n1 = x.size();
	const size_t n2 = x.front().size();
	double res = 0.0;
	size_t p = off;
	for(size_t i = 0; i < n1; ++i){
		for(size_t j = 0; j < n2; ++j)
			res += x[i][j] * w[p++];
	}
	return sigmoid(res+w[p]);
}

std::vector<std::vector<double>> FCNode1D::gradient(
	std::vector<double>& grad, const std::vector<std::vector<double>>& x,
	const std::vector<double>& w, const double& y, const double& pre)
{
	const size_t n1 = x.size();
	const size_t n2 = x.front().size();
	//assert(y.size() == 1 && pre.size() == 1);
	//const double f = pre[0] * y[0];
	const double d = sigmoid_derivative(0.0, y);
	const double f = pre * d;
	std::vector<std::vector<double>> pg(n1, vector<double>(n2));
	size_t p = off;
	for(size_t i = 0; i < n1; ++i){
		for(size_t j = 0; j < n2; ++j){
			grad[p] = x[i][j] * f; // pre * dy/dw
			pg[i][j] = w[p] * f; // pre * dy/dx
			++p;
		}
	}
	grad[p] = f; // the constant offset
	return pg;
}
