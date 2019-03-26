#pragma once
#include "DataPoint.h"
#include <string>
#include <unordered_set>

DataPoint parseLine(const std::string& line, const std::string& sepper,
	const std::unordered_set<int>& xIds, const std::unordered_set<int>& yIds, const bool appOne);

void parseLineNMF(const std::string& line, const std::string& sepper, 
	std::vector<DataPoint>& data, const int xi, const bool appOne);

void addData(std::vector<DataPoint>& data, const int xi, const int yi, double lb);