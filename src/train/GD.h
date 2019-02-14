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
};

