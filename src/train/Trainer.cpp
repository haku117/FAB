#include "Trainer.h"
#include "logging/logging.h"
// #include <iostream>
using namespace std;

Trainer::Trainer(){
	this->pm = nullptr;
	this->pd = nullptr;
	// LOG(INFO) << "inital trainer pointers";
}

void Trainer::bindModel(Model* pm){
	this->pm = pm;
	//VLOG(3) << "bind model " << pm->kernelName(); 
	// if(this->pd != nullptr) { //&& this->pm->getKernel()->lengthState() != 0){
	// 	initState();
	// }
}

void Trainer::bindDataset(const DataHolder* pd){
	// VLOG(2) << "Trainer bind dataset data[0]: " << pd->get(0).x << " -> " << pm->getKernel()->lengthState(); 
	// VLOG(3) << "pm name " << this->pm->kernelName();
	this->pd = pd;
	// if(this->pm != nullptr) { // && pm->getKernel()->lengthState() != 0){
	// 	initState();
	// }
}

double Trainer::loss() const {
	double res = 0;
	size_t n = pd->size();
	// cout << "start trainer loss" << endl;
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

// void Trainer::setZ(std::vector<std::vector<int> > z) {
// 	return;
// }

// std::vector<std::vector<int> > Trainer::getZ() const{
// 	std::vector<std::vector<int> > a;
// 	return a;
// }

void Trainer::initState (int dim){
	DLOG(INFO) << "Trainer init State ??";
	return;
}