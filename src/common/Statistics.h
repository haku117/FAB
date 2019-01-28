#pragma once
#include <cstddef> // for size_t on GCC

struct Statistics {
	// network
	size_t n_net_send, n_net_recv;
	size_t b_net_send, b_net_recv;
	double t_net_send, t_net_recv;
	double t_data_serial, t_data_deserial;
	// delta
	size_t n_dlt_send, n_dlt_recv;
	double t_dlt_calc, t_dlt_wait;
	// parameter
	size_t n_par_send, n_par_recv;
	double t_par_calc, t_par_wait;

	Statistics();
};
