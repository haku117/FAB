#include "norm.h"

using namespace std;

double l1norm(const std::vector<double>& vec) {
	double r = 0.0;
	for (auto& v : vec)
		r += (v >= 0.0 ? v : -v);
	return r;
}

double l2norm(const std::vector<double>& vec) {
	double r = 0.0;
	for (auto& v : vec)
		r += v * v;
	return r;
}

double l1norm_diff(const std::vector<double>& vec1, const std::vector<double>& vec2)
{
	double r = 0.0;
	for (size_t i = 0; i < vec1.size();++i) {
		double v = vec1[i] - vec2[i];
		r += (v >= 0.0 ? v : -v);
	}
	return r;
}

double l2norm_diff(const std::vector<double>& vec1, const std::vector<double>& vec2)
{
	double r = 0.0;
	for (size_t i = 0; i < vec1.size(); ++i) {
		double v = vec1[i] - vec2[i];
		r += v * v;
	}
	return r;
}
