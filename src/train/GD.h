#pragma once
#include "Trainer.h"

class GD : public Trainer
{
	double rate;

public:
	GD();
	void setRate(const double rate);
	double getRate() const;

	virtual std::vector<double> batchDelta(
		const size_t start, const size_t cnt, const bool avg = true);
	virtual std::pair<size_t, std::vector<double>> batchDelta(
		std::atomic<bool>& cond, const size_t start, const size_t cnt, const bool avg = true);
	///// temprate remove const for 2 functions
	virtual std::pair<size_t, std::vector<double>> batchDelta(
		std::atomic<bool>& cond, const size_t start, const size_t cnt, const double sleep, 
				const double interval, const bool avg = true);

	virtual std::pair<size_t, std::vector<double>> batchDeltaPipe(std::atomic<bool>& cond, 
		const size_t start, const size_t cnt, const size_t blk, const std::vector<int> blkSize, 
		const bool avg = true) {};

	// virtual void initState(int dim = 0);
};

