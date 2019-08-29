#pragma once
#include <utility>
#include <string>
#include <vector>

std::pair<int, int> getScreenSize();
bool beTrueOption(const std::string& str);

std::vector<int> getIntList(const std::string & str, const std::string& sepper = " ,");
std::vector<double> getDoubleList(const std::string & str, const std::string& sepper = " ,");
std::vector<std::string> getStringList(const std::string & str, const std::string& sepper = " ,");

// binary: whether to use 2^10=1024 or 1000
int stoiKMG(const std::string & str, const bool binary = false);
size_t stoulKMG(const std::string & str, const bool binary = false);

void accumuteDeltaSave(std::vector<double>& delta, std::vector<double>& d);
std::vector<int> parseParam(const std::string& param);
int str2int(const std::string& token);

double l1norm0(const std::vector<double>& dd);
double l2norm0(const std::vector<double>& dd);

// for LDA
double digamma(double x);
double log_sum(double log_a, double log_b);
std::vector<double> ss2param(const std::vector<double>& ss, int k);

// for GMM
// double prob(std::vector<double>& xi, )
