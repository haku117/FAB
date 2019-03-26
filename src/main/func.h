#pragma once
#include <string>
#include <vector>
#include <utility>

std::pair<double, std::vector<double>> parseRecordLine(const std::string& line);

std::pair<std::vector<double>, std::vector<double>> parseRecordLineIter(const std::string& line);

