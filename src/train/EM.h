#pragma once
#include "Trainer.h"

class EM : public Trainer
{
	double rate;
	/// local stats for each data point
	std::vector<std::vector<int> > z;
	// std::vector<std::vector<double> > localParam;

public:
	EM();
	void setRate(const double rate);
	double getRate() const;

	// void setZ(std::vector<std::vector<int> > z);
	// std::vector<std::vector<int> > getZ() const;

	void initState(int dim, int nnx = 0);

	virtual std::vector<double> batchDelta(
		const size_t start, const size_t cnt, const bool avg = true);
	virtual std::pair<size_t, std::vector<double>> batchDelta(
		std::atomic<bool>& cond, const size_t start, const size_t cnt, const bool avg = true);
	///// temprate remove const for 2 functions

	virtual std::pair<size_t, std::vector<double>> batchDeltaPipe(std::atomic<bool>& cond, 
		const size_t start, const size_t cnt, const size_t blk, const std::vector<int> blkSize, 
		const bool avg = true);
	// void EM::accumuteDeltaSave(std::vector<double>& delta, std::vector<double>& d);
};

