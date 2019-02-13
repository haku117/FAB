#pragma once
#include <vector>

double sigmoid(double x);
double sigmoid_derivative(double x);
double sigmoid_derivative(double x, double y);

double relu(double x);
double relu_derivative(double x);
double relu_derivative(double x, double y);

//double tanh(double x);
double tanh_derivative(double x);
double tanh_derivative(double x, double y);

double arctan(double x);
double arctan_derivative(double x);
double arctan_derivative(double x, double y);

double softplus(double x);
double softplus_derivative(double x);
double softplus_derivative(double x, double y);


std::vector<double> softmax(const std::vector<double>& x);
