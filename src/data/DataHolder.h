#pragma once
#include "DataPoint.h"
#include <vector>
#include <string>

class DataHolder {
	std::vector<DataPoint> data;
	size_t npart; // total number of parts
	size_t pid; // part id

	bool appendOne;
	size_t nx; // length of x
	size_t ny; // length of y
public:
	// <appOne>: append a constant value (one) to each x
	// <nparts> <localid>: used for distributed case
	DataHolder(const bool appOne = true, const size_t nparts = 1, const size_t localid = 0);
	size_t xlength() const;
	size_t ylength() const;

	// give the column id of y and the skipped ones, the rest are x. id starts from 0
	// throw exceptions if something wrong
	void load(const std::string& fpath, const std::string& sepper,
		const std::vector<int> skips, const std::vector<int>& yIds,
		const bool header, const bool onlyLocalPart = false, const int pp = 1);

	void loadNMF(const std::string& fpath, const std::string& sepper,
		// const std::vector<int> skips, const std::vector<int>& yIds,
		const std::string& param,
		const bool header, const bool onlyLocalPart = false);

	void loadLDA(const std::string& fpath, const std::string& sepper, const int pp = 1);

	void add(const std::vector<double>& x, const std::vector<double>& y);
	void add(std::vector<double>&& x, std::vector<double>&& y);

	size_t size() const {
		return data.size();
	}
	const DataPoint& get(const size_t idx) const {
		return data[idx];
	}

	// normalize to [-1, 1]
	void normalize(const bool onY);
};
