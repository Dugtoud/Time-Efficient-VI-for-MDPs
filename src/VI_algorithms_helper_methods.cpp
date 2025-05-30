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

//Returns the maximum absolute difference between entries in the two vectors
//pass the arguments by reference for efficiency, use const?
double abs_max_diff(double V_one[], double V_two[], int S){
		double abs_max = double(0);
		for (int i = 0; i < S; ++i) {
				abs_max = max(abs_max, abs(V_one[i] - V_two[i]));	
		}
		return abs_max;
}


//define argument references as const, as they are not changed
double sum_of_mult_nonzero_only(const vector<double> &V_one, double V_two[], const vector<int> &non_zero_transition_states){
		double cum_sum = double(0);
		for (int s : non_zero_transition_states) {
				cum_sum += 	(V_one[s] * V_two[s]);
		}
		return cum_sum;
}

//define argument references as const, as they are not changed
double sum_of_mult(const vector<double> &V_one, double V_two[]){
		double cum_sum = double(0);
		for (int i = 0; i < V_one.size(); ++i) {
				cum_sum += 	(V_one[i] * V_two[i]);
		}
		return cum_sum;
}

double find_max_R(const R_type &R){
		double max_R = numeric_limits<double>::min();
		for (int i = 0; i < R.size(); i++) {
				for(int j = 0; j < R[i].size(); j++){
						max_R = max(max_R, R[i][j]);	
				}
		}
		return max_R;
}

vector<double> find_max_R_for_each_state(const R_type &R){
		int S = R.size();
		vector<double> max_R_for_each_state(S);

		//For each state
		for (int i = 0; i < S; i++) {
				double max_R = numeric_limits<double>::min();
				for(int j = 0; j < R[i].size(); j++){
						max_R = max(max_R, R[i][j]);	
				}
				max_R_for_each_state[i] = max_R;
		}
		return max_R_for_each_state;
}

tuple<double, double, vector<double>> find_all_r_values(const R_type &R){
		int S = R.size();
		vector<double> max_R_for_each_state(S);
		double r_star_min = numeric_limits<double>::max();
		double r_star_max = numeric_limits<double>::min();

		//For each state
		for (int i = 0; i < S; i++) {
				//find max reward in each state
				double max_R = numeric_limits<double>::min();

				for(int j = 0; j < R[i].size(); j++){
						max_R = max(max_R, R[i][j]);	
				}
				
				//test if it is the r_* value among states so far
				r_star_min = min(r_star_min, max_R);

				//test if largest among states so far
				r_star_max = max(r_star_max, max_R);

				max_R_for_each_state[i] = max_R;
		}

		return make_tuple(r_star_min, r_star_max, max_R_for_each_state);
}

double find_min_R(const R_type &R){
		double min_R = numeric_limits<double>::max();
		for (int i = 0; i < R.size(); i++) {
				for(int j = 0; j < R[i].size(); j++){
						min_R = min(min_R, R[i][j]);	
				}
		}
		return min_R;
}

vector<double> V_upper_lower_average(double V_one[], double V_two[], int S){
		vector<double> answer(S,0);
		for (int i = 0; i < S; ++i) {
				double average_of_index_i = (V_one[i] + V_two[i]) / double(2);
				answer[i] = average_of_index_i;
		}
		return answer;
}

//Find the maximum action to have this as the maximum action to keep indicies for in every state	
int find_max_A(const A_type &A){
		int max_A = 0;
		for (int i = 0; i < A.size(); i++) {
				for(int j = 0; j < A[i].size(); j++){
						max_A = max(max_A, A[i][j]);	
				}
		}
		return max_A;
}

void does_heap_and_indicies_match(heap_of_pairs_type heap, int indicies[], int A_max){
		for(int i = 0; i < A_max; i++){
				if (indicies[i] != -1){
						if (heap[indicies[i]].second != i){
								printf("THEY DO NOT MATCH!\n");
						}
				}
		}
}

//check if same size first!
void are_heaps_synced(heap_of_pairs_type &max_heap, heap_of_pairs_type &min_heap){
		if (max_heap.size() != min_heap.size()){
				printf("THEY ARE NOT THE SAME SIZE\n");
				printf("max\n");
				print_heap(max_heap);
				printf("min\n");
				print_heap(min_heap);
		}
		for(auto p_max : max_heap){
				for(auto p_min : min_heap){
						if (p_max.second == p_min.second){
								if (p_max.first != p_min.first){
										printf("THE TWO HEAPS ARE NOT SYNCED!\n");
								}
						}
				}
		}

}

A_type copy_A(const A_type &A){
		A_type A_copy;
		for(auto a_s : A){
				A_copy.push_back(a_s);
		}
		return A_copy;
}

double abs_max_diff_vectors(const V_result_type &V_one, const V_result_type &V_two){
		double abs_max = double(0);
		for (int i = 0; i < V_one.size(); ++i) {
				abs_max = max(abs_max, abs(V_one[i] - V_two[i]));	
		}
		return abs_max;
}
