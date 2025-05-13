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
#include "pretty_printing_MDP.h"
#include "MDP_generation.h"
#include "VI_algorithms_helper_methods.h"
#include "VI_algorithm.h"
#include "BVI_algorithm.h"
#include "VIAE_algorithm.h"
#include "VIAEH_algorithm.h"
#include "VIH_algorithm.h"
#include "experiments.h"

using namespace std;
using namespace std::chrono;

V_type value_iteration(S_type S, R_type R, A_type A, P_type P, double gamma, double epsilon){
		
		//Find the maximum reward in the reward table
		auto [r_star_min, r_star_max, r_star_values] = find_all_r_values(R);

		//2. Improved Lower Bound
		double** V = new double*[2];
		for(int i = 0; i < 2; ++i) {
				V[i] = new double[S];
		}

		for(int s = 0; s < S; s++) {
				V[0][s] = (gamma / (1.0 - gamma)) * r_star_min + r_star_values[s];
				V[1][s] = 1.0;
		}

		//record actions eliminated in each iteration, where a pair is (state, action)
		//push empty vector for 0-index. Iterations start with 1
		vector<vector<pair<int, int>>> actions_eliminated;
		actions_eliminated.push_back(vector<pair<int, int>>());

		//keep track of work done in each iteration in microseconds
		//start from iteration 1
		vector<microseconds> work_per_iteration(1);

		//keep count of number of iterations
		int iterations = 0;
		
		while (abs_max_diff(V[0], V[1], S) > ((epsilon * (1.0 - gamma)) / gamma)){

				//Increment iteration counter i
				iterations++;	
				
				//Record actions eliminated in this iteration over all states
				vector<pair<int, int>> actions_eliminated_in_iteration;
			
				//begin timing of this iteration
				auto start_of_iteration = high_resolution_clock::now();
				
				//If i is even, then (i & 1) is 0, and the one to change is V[0]
				double *V_current_iteration = V[(iterations & 1)];
				double *V_previous_iteration = V[1 - (iterations & 1)];

				//for all states in each iteration
				for (int s = 0; s < S; s++) {
						//TODO if non-negative rewards, 0 is a lower bound of the maximization. to be changed if we want negative rewards
						V_current_iteration[s] = double(0);

						//ranged for loop over all actions in the action set of state s
						for (auto a : A[s]) {
								auto& [P_s_a, P_s_a_nonzero] = P[s][a];
								double R_s_a = R[s][a] + gamma * sum_of_mult_nonzero_only(P_s_a, V_previous_iteration, P_s_a_nonzero); 
								if (R_s_a > V_current_iteration[s]) {
										V_current_iteration[s] = R_s_a;
								}
						}
				}
				
				//end timing of this iteration and record it in work vector
				auto end_of_iteration = high_resolution_clock::now();
				auto duration_of_iteration = duration_cast<microseconds>(end_of_iteration - start_of_iteration);
				work_per_iteration.push_back(duration_of_iteration);
				actions_eliminated.push_back(move(actions_eliminated_in_iteration));
		}
		vector<double> result(V[(iterations & 1)], V[(iterations & 1)] + S);
		V_type result_tuple = make_tuple(result, iterations, work_per_iteration, actions_eliminated);
	
		//DEALLOCATE MEMORY
		for(int i = 0; i < 2; ++i) {
				delete [] V[i];
		}
		delete [] V;
	
		return result_tuple;
}
