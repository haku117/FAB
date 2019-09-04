#pragma once
#include "common/ConfData.h"

struct Option{
	ConfData conf;

	bool parse(int argc, char* argv[], const size_t nWorker);
	void showUsage() const;

private:
	bool processMode();
	bool processDataset();
	bool processAlgorithm();
	bool processOptimizer();

	struct Impl;
	Impl* pimpl = nullptr;
};
