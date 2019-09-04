#pragma once
#include <string>
#include <vector>

struct Option{
	//std::string prefix;
	std::string fnData;
	std::string fnOutput;
	std::vector<int> idSkip;
	std::vector<int> idY;
	bool header;
	bool doNormalize;

	std::string mode;
	bool fabWait; // force fab wait for its gradient reply before continue
	std::string algorighm;
	std::string algParam;
	std::string straggler;
	int reptr, dataSize;

	double lrate; // learning rate
	size_t nw; // number of workers
	size_t batchSize;

	// condition for archive result and write log
	size_t arvIter; // every n iterations
	double arvTime; // every t seconds
	int logIter;
	
	// termination condition
	size_t tcIter; // maximum iteration
	double tcTime; // maximum training time
	double tcDiff; // minimum improvement cross iterations

	bool parse(int argc, char* argv[], const size_t nWorker);
	void showUsage() const;

private:
	bool preprocessMode();
	bool processAlgorithm();

	struct Impl;
	//Impl* pimpl;
};
