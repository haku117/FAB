#include "Trainer.h"
// #include "logging/logging.h"
using namespace std;

Trainer::Trainer(){
	this->pm = nullptr;
	this->pd = nullptr;
}

void Trainer::bindModel(Model* pm){
	this->pm = pm;
	if(pd != nullptr && pm->getKernel()->lengthState() != 0){
		initState();
	}
}

void Trainer::bindDataset(const DataHolder* pd){
	this->pd = pd;
	if(pm != nullptr && pm->getKernel()->lengthState() != 0){
		initState();
	}
}

void Trainer::init(){
	
}

/// initialize the local state Z
void Trainer::initState(){
	int xlen = pd->size();
	int dim = pm->getKernel()->lengthState();
	std::vector<std::vector<int> > matrix(xlen, std::vector<int>(dim, -1));
	z = move(matrix);
	// VLOG(1) << "initalize local state Z: " << z.size(); 
}

double Trainer::loss() const {
	double res = 0;
	size_t n = pd->size();
	for(size_t i = 0; i < n; ++i){
		res += pm->loss(pd->get(i));
	}
	return res / static_cast<double>(n);
}

void Trainer::train(const size_t start, const size_t cnt)
{
	vector<double> delta = batchDelta(start, cnt != 0 ? cnt : pd->size(), true);
	applyDelta(delta, 1.0);
}

size_t Trainer::train(std::atomic<bool>& cond, const size_t start, const size_t cnt)
{
	pair<size_t, vector<double>> res = batchDelta(cond, start, cnt != 0 ? cnt : pd->size(), true);
	applyDelta(res.second, 1.0);
	return res.first;
}

void Trainer::applyDelta(const vector<double>& delta, const double factor)
{
	pm->accumulateParameter(delta, factor);
}
