#ifndef VIAEH_ALGORITHM_MAXMIN_HEAP_H
#define VIAEH_ALGORITHM_MAXMIN_HEAP_H

#include <algorithm>
#include <chrono>
#include <tuple>
#include <random>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>

#include "MDP_type_definitions.h"

using namespace std;

V_type value_iteration_action_elimination_heaps_max_min_heap(S_type S, R_type R, A_type A, P_type P, double gamma, double epsilon);
#endif
