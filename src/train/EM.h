#pragma once
#include "Trainer.h"

class EM : public Trainer
{
	double rate;
	/// local stats for each data point
	std::vector<std::vector<int> > z;

public:
	EM();
	void setRate(const double rate);
	double getRate() const;

	// void setZ(std::vector<std::vector<int> > z);
	// std::vector<std::vector<int> > getZ() const;

	void initState();

	virtual std::vector<double> batchDelta(
		const size_t start, const size_t cnt, const bool avg = true);
	virtual std::pair<size_t, std::vector<double>> batchDelta(
		std::atomic<bool>& cond, const size_t start, const size_t cnt, const bool avg = true);
	///// temprate remove const for 2 functions
};

