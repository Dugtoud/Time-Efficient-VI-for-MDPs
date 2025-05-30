#include <algorithm>
#include <chrono>
#include <ctime>
#include <tuple>
#include <random>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <math.h>

#include "MDP_type_definitions.h"
#include "pretty_printing_MDP.h"
#include "MDP_generation.h"
#include "VI_algorithms_helper_methods.h"
#include "VI_algorithm.h"
#include "VIU_algorithm.h"
#include "BVI_algorithm.h"
#include "VIAE_algorithm.h"
#include "VIAE_algorithm_improved_bounds.h"
#include "VIAE_algorithm_old_bounds.h"
#include "VIAEH_algorithm.h"
#include "VIAEH_algorithm_no_pointers.h"
#include "VIAEH_algorithm_maxmin_heap.h"
#include "VIAEH_algorithm_lower_bound_approx.h"
#include "VIAEH_algorithm_lazy_update.h"
#include "VIAEH_algorithm_set.h"
#include "VIH_algorithm.h"
#include "experiments.h"
#include "stopping_criteria_plot.h"
#include "top_action_change_plot.h"
#include "VIH_actions_touched.h"
#include "BAO_algorithm.h"
#include "VIH_algorithm_custom_heaps.h"

using namespace std;
using namespace std::chrono;

//GENERAL HELPER METHOD TO WRITE DATA TO FILE
void write_stringstream_to_file(ostringstream& string_stream, ofstream& output_stream, string file_name){
		output_stream.open(file_name);
		if (output_stream.is_open()){
				printf("opened file: success\n");
				output_stream << string_stream.str();
				output_stream.close();
		} else {
				printf("opened file: fail\n");
		}
}

//VARYING ACTION PROBABILITY EXPERIMENT
void write_meta_data_to_dat_file_action_prob(ostringstream& string_stream, int S, int A_num, double epsilon, double gamma, double upper_reward, double non_zero_transition, double action_prob_starting_value, double action_prob_finishing_value, double action_prob_increment){
		time_t time_now = time(0);
		string_stream << "# META DATA" << endl;
		string_stream << "# " << endl;
		string_stream << "# " << "experiment run at: " << ctime(&time_now);
		string_stream << "# " << endl;
		string_stream << "# " << "gamma = " << gamma << endl;
		string_stream << "# " << "epsilon = " << epsilon << endl;
		string_stream << "# " << "S = " << S << endl;
		string_stream << "# " << "A = " << A_num << endl;
		string_stream << "# " << "non_zero_transition = " << non_zero_transition << endl;
		string_stream << "# " << "upper_reward = " << upper_reward << endl;
		string_stream << "# " << endl;
		string_stream << "# VARIABLE IN THIS EXPERIMENT" << endl;
		string_stream << "# " << "action_prob is varied from " << action_prob_starting_value << " to " << action_prob_finishing_value << " with " << action_prob_increment << " increment" << endl;
		string_stream << "# " << endl;
		string_stream << "# ACTUAL DATA" << endl;
		string_stream << "# action_prob | microseconds" << endl;
}

void create_data_tables_action_prob(string filename, int S, int A_num, double epsilon, double gamma, double upper_reward, double non_zero_transition){

		//the stringstreams to create the test for the files
		ostringstream stringstream_BVI;
		ostringstream stringstream_VI;
		ostringstream stringstream_VIU;
		ostringstream stringstream_VIH;
		ostringstream stringstream_VIAE;
		ostringstream stringstream_VIAEH;

		//the file output objects
		ofstream output_stream_BVI;
		ofstream output_stream_VI;
		ofstream output_stream_VIU;
		ofstream output_stream_VIH;
		ofstream output_stream_VIAE;
		ofstream output_stream_VIAEH;
		
		//set the name of the file to write to
		string file_name_BVI = "data_tables/action_prob/" + filename + "_BVI.dat"; 
		string file_name_VI =  "data_tables/action_prob/" + filename + "_VI.dat";
		string file_name_VIU = "data_tables/action_prob/" + filename + "_VIU.dat";
		string file_name_VIH = "data_tables/action_prob/" + filename + "_VIH.dat";
		string file_name_VIAE = "data_tables/action_prob/" + filename + "_VIAE.dat";
		string file_name_VIAEH = "data_tables/action_prob/" + filename + "_VIAEH.dat";

		double action_prob_starting_value = 0.10;
		double action_prob_finishing_value = 1.0;
		double action_prob_increment = 0.05;

		//write meta data to all stringstreams as first in their respective files
		write_meta_data_to_dat_file_action_prob(stringstream_VI, S, A_num, epsilon, gamma, upper_reward, non_zero_transition, action_prob_starting_value, action_prob_finishing_value, action_prob_increment);
		write_meta_data_to_dat_file_action_prob(stringstream_VIU, S, A_num, epsilon, gamma, upper_reward, non_zero_transition, action_prob_starting_value, action_prob_finishing_value, action_prob_increment);
		write_meta_data_to_dat_file_action_prob(stringstream_VIH, S, A_num, epsilon, gamma, upper_reward, non_zero_transition, action_prob_starting_value, action_prob_finishing_value, action_prob_increment);
		write_meta_data_to_dat_file_action_prob(stringstream_BVI, S, A_num, epsilon, gamma, upper_reward, non_zero_transition, action_prob_starting_value, action_prob_finishing_value, action_prob_increment);
		write_meta_data_to_dat_file_action_prob(stringstream_VIAE, S, A_num, epsilon, gamma, upper_reward, non_zero_transition, action_prob_starting_value, action_prob_finishing_value, action_prob_increment);
		write_meta_data_to_dat_file_action_prob(stringstream_VIAEH, S, A_num, epsilon, gamma, upper_reward, non_zero_transition, action_prob_starting_value, action_prob_finishing_value, action_prob_increment);

		for (double action_prob = action_prob_starting_value; action_prob <= action_prob_finishing_value; action_prob = action_prob + action_prob_increment){
				
				//status message of experiment
				printf("Beginning iteration action_prob = %f\n", action_prob);

				//GENERATE THE MDP FROM CURRENT TIME SEED
				int seed = time(0);
				auto MDP = generate_random_MDP_with_variable_parameters(S, A_num, action_prob, non_zero_transition, upper_reward, seed);
				R_type R = get<0>(MDP);
				A_type A = get<1>(MDP);
				P_type P = get<2>(MDP);

				//VI testing
				//TODO: make method that takes arguments and performs the writing of data, takes filename, MDP etc. as argument
				A_type A1 = copy_A(A);
				auto start_VI = high_resolution_clock::now();

				V_type V_approx_solution_tuple = value_iteration(S, R, A1, P, gamma, epsilon);
                vector<double> V_approx_solution = get<0>(V_approx_solution_tuple);

				auto stop_VI = high_resolution_clock::now();
				auto duration_VI = duration_cast<microseconds>(stop_VI - start_VI);

				stringstream_VI << to_string(action_prob) << " " << duration_VI.count() << endl;

				//VIU testing
				A_type A6 = copy_A(A);
				auto start_VIU = high_resolution_clock::now();

				V_type V_approx_solution_upper_tuple = value_iteration_upper(S, R, A6, P, gamma, epsilon);
				vector<double> V_approx_solution_upper = get<0>(V_approx_solution_upper_tuple);

				auto stop_VIU = high_resolution_clock::now();
				auto duration_VIU = duration_cast<microseconds>(stop_VIU - start_VIU);

				stringstream_VIU << to_string(action_prob) << " " << duration_VIU.count() << endl;

				//VIH testing
				A_type A2 = copy_A(A);
				auto start_VIH = high_resolution_clock::now();

				V_type V_heap_approx_tuple = value_iteration_with_heap(S, R, A2, P, gamma, epsilon);
				vector<double> V_heap_approx = get<0>(V_heap_approx_tuple);

				auto stop_VIH = high_resolution_clock::now();
				auto duration_VIH = duration_cast<microseconds>(stop_VIH - start_VIH);

				stringstream_VIH << to_string(action_prob) << " " << duration_VIH.count() << endl;

				//BVI
				A_type A3 = copy_A(A);
				auto start_BVI = high_resolution_clock::now();

				V_type V_bounded_approx_solution_tuple = bounded_value_iteration(S, R, A3, P, gamma, epsilon);
				vector<double> V_bounded_approx_solution = get<0>(V_bounded_approx_solution_tuple);

				auto stop_BVI = high_resolution_clock::now();
				auto duration_BVI = duration_cast<microseconds>(stop_BVI - start_BVI);
				
				stringstream_BVI << to_string(action_prob) << " " << duration_BVI.count() << endl;

				//VIAE
				A_type A4 = copy_A(A);
				auto start_VIAE = high_resolution_clock::now();

				V_type V_AE_approx_solution_tuple = value_iteration_action_elimination(S, R, A4, P, gamma, epsilon);
				vector<double> V_AE_approx_solution = get<0>(V_AE_approx_solution_tuple);

				auto stop_VIAE = high_resolution_clock::now();
				auto duration_VIAE = duration_cast<microseconds>(stop_VIAE - start_VIAE);

				stringstream_VIAE << to_string(action_prob) << " " << duration_VIAE.count() << endl;

				//VIAEH
				A_type A5 = copy_A(A);
				auto start_VIAEH = high_resolution_clock::now();

				V_type V_AE_H_approx_solution_tuple = value_iteration_action_elimination_heaps(S, R, A5, P, gamma, epsilon);
				vector<double> V_AE_H_approx_solution = get<0>(V_AE_H_approx_solution_tuple);

				auto stop_VIAEH = high_resolution_clock::now();
				auto duration_VIAEH = duration_cast<microseconds>(stop_VIAEH - start_VIAEH);

				stringstream_VIAEH << to_string(action_prob) << " " << duration_VIAEH.count() << endl;

				//They should in theory all be epsilon from true value, and therefore, at most be 2 * epsilon from each other
				if (abs_max_diff_vectors(V_approx_solution, V_heap_approx) > (2 * epsilon)){
						printf("DIFFERENCE\n");
				}
				if (abs_max_diff_vectors(V_AE_approx_solution, V_AE_H_approx_solution) > (2 * epsilon)){
						printf("DIFFERENCE\n");
				}
				if (abs_max_diff_vectors(V_bounded_approx_solution, V_approx_solution_upper) > (2 * epsilon)){
						printf("DIFFERENCE\n");
				}

		}

		//WRITE ALL DATA TO THEIR RESPECTVIE FILES	
		write_stringstream_to_file(stringstream_VI, output_stream_VI, file_name_VI);
		write_stringstream_to_file(stringstream_VIU, output_stream_VIU, file_name_VIU);
		write_stringstream_to_file(stringstream_BVI, output_stream_BVI, file_name_BVI);
		write_stringstream_to_file(stringstream_VIH, output_stream_VIH, file_name_VIH);
		write_stringstream_to_file(stringstream_VIAE, output_stream_VIAE, file_name_VIAE);
		write_stringstream_to_file(stringstream_VIAEH, output_stream_VIAEH, file_name_VIAEH);

}

//VARYING NUMBER OF STATES EXPERIMENTS
void write_meta_data_to_dat_file_number_of_states(ostringstream& string_stream, int A_num, double epsilon, double gamma, double upper_reward, double non_zero_transition, double action_prob, int S_starting_value, int S_finishing_value, int S_increment){
		time_t time_now = time(0);
		string_stream << "# META DATA" << endl;
		string_stream << "# " << endl;
		string_stream << "# " << "experiment run at: " << ctime(&time_now);
		string_stream << "# " << endl;
		string_stream << "# " << "gamma = " << gamma << endl;
		string_stream << "# " << "epsilon = " << epsilon << endl;
		string_stream << "# " << "A = " << A_num << endl;
		string_stream << "# " << "non_zero_transition = " << non_zero_transition << endl;
		string_stream << "# " << "upper_reward = " << upper_reward << endl;
		string_stream << "# " << "action_prob = " << action_prob << endl;
		string_stream << "# " << endl;
		string_stream << "# VARIABLE IN THIS EXPERIMENT" << endl;
		string_stream << "# " << "Number of states, S, is varied from " << S_starting_value << " to " << S_finishing_value << " with " << S_increment << " increment" << endl;
		string_stream << "# " << endl;
		string_stream << "# ACTUAL DATA" << endl;
		string_stream << "# number of states | microseconds" << endl;
}

void create_data_tables_number_of_states(string filename, int S_max, int A_num, double epsilon, double gamma, double upper_reward, double non_zero_transition){

		//the stringstreams to create the test for the files
		ostringstream stringstream_BVI;
		ostringstream stringstream_VI;
		ostringstream stringstream_VIU;
		ostringstream stringstream_VIH;
		ostringstream stringstream_VIAE;
		ostringstream stringstream_VIAEH;
		ostringstream stringstream_VIAEHL;
		ostringstream stringstream_BAO;

		//the file output objects
		ofstream output_stream_BVI;
		ofstream output_stream_VI;
		ofstream output_stream_VIU;
		ofstream output_stream_VIH;
		ofstream output_stream_VIAE;
		ofstream output_stream_VIAEH;
		ofstream output_stream_VIAEHL;
		ofstream output_stream_BAO;
		
		//set the name of the file to write to
		string file_name_BVI = "data_tables/number_of_states/" + filename + "_BVI.dat"; 
		string file_name_VI = "data_tables/number_of_states/" + filename + "_VI.dat";
		string file_name_VIU = "data_tables/number_of_states/" + filename + "_VIU.dat";
		string file_name_VIH = "data_tables/number_of_states/" + filename + "_VIH.dat";
		string file_name_VIAE = "data_tables/number_of_states/" + filename + "_VIAE.dat";
		string file_name_VIAEH = "data_tables/number_of_states/" + filename + "_VIAEH.dat";
		string file_name_VIAEHL = "data_tables/number_of_states/" + filename + "_VIAEHL.dat";
		string file_name_BAO = "data_tables/number_of_states/" + filename + "_BAO.dat";

		//The varying parameters
		int S_starting_value = 50;
		int S_finishing_value = S_max;
		int S_increment = 50;

		//hardcoded parameter
		double action_prob = 1.0;

		//write meta data to all stringstreams as first in their respective files
		
		write_meta_data_to_dat_file_number_of_states(stringstream_VI, A_num, epsilon, gamma, upper_reward, non_zero_transition, action_prob, S_starting_value, S_finishing_value, S_increment);
		write_meta_data_to_dat_file_number_of_states(stringstream_VIU, A_num, epsilon, gamma, upper_reward, non_zero_transition, action_prob, S_starting_value, S_finishing_value, S_increment);
		write_meta_data_to_dat_file_number_of_states(stringstream_BVI, A_num, epsilon, gamma, upper_reward, non_zero_transition, action_prob, S_starting_value, S_finishing_value, S_increment);
		write_meta_data_to_dat_file_number_of_states(stringstream_VIH, A_num, epsilon, gamma, upper_reward, non_zero_transition, action_prob, S_starting_value, S_finishing_value, S_increment);
		write_meta_data_to_dat_file_number_of_states(stringstream_VIAE, A_num, epsilon, gamma, upper_reward, non_zero_transition, action_prob, S_starting_value, S_finishing_value, S_increment);
		write_meta_data_to_dat_file_number_of_states(stringstream_VIAEH, A_num, epsilon, gamma, upper_reward, non_zero_transition, action_prob, S_starting_value, S_finishing_value, S_increment);

		for (int S = S_starting_value; S <= S_finishing_value; S = S + S_increment){
				
				printf("Beginning iteration S = %d\n", S);

				//GENERATE THE MDP
				int seed = time(0);
				auto MDP = generate_random_MDP_normal_distributed_rewards(S, A_num, action_prob, S, seed, 1000, 10);
				//auto MDP = generate_random_MDP_with_variable_parameters(S, A_num, action_prob, non_zero_transition, upper_reward, seed);
				R_type R = get<0>(MDP);
				A_type A = get<1>(MDP);
				P_type P = get<2>(MDP);

				//VI testing
				//TODO: make method that takes arguments and performs the writing of data, takes filename, MDP etc. as argument
				A_type A1 = copy_A(A);
				auto start_VI = high_resolution_clock::now();

				V_type V_approx_solution_tuple = value_iteration(S, R, A1, P, gamma, epsilon);
                vector<double> V_approx_solution = get<0>(V_approx_solution_tuple);

				auto stop_VI = high_resolution_clock::now();
				auto duration_VI = duration_cast<microseconds>(stop_VI - start_VI);

				stringstream_VI << to_string(S) << " " << duration_VI.count() << endl;

				//VIU testing
				A_type A6 = copy_A(A);
				auto start_VIU = high_resolution_clock::now();

				V_type V_approx_solution_upper_tuple = value_iteration_upper(S, R, A6, P, gamma, epsilon);
				vector<double> V_approx_solution_upper = get<0>(V_approx_solution_upper_tuple);

				auto stop_VIU = high_resolution_clock::now();
				auto duration_VIU = duration_cast<microseconds>(stop_VIU - start_VIU);

				stringstream_VIU << to_string(S) << " " << duration_VIU.count() << endl;

				//VIH testing
				A_type A2 = copy_A(A);
				auto start_VIH = high_resolution_clock::now();

				V_type V_heap_approx_tuple = value_iteration_with_heap(S, R, A2, P, gamma, epsilon);
				vector<double> V_heap_approx = get<0>(V_heap_approx_tuple);

				auto stop_VIH = high_resolution_clock::now();
				auto duration_VIH = duration_cast<microseconds>(stop_VIH - start_VIH);

				stringstream_VIH << to_string(S) << " " << duration_VIH.count() << endl;

				//BVI
				A_type A3 = copy_A(A);
				auto start_BVI = high_resolution_clock::now();

				V_type V_bounded_approx_solution_tuple = bounded_value_iteration(S, R, A3, P, gamma, epsilon);
				vector<double> V_bounded_approx_solution = get<0>(V_bounded_approx_solution_tuple);

				auto stop_BVI = high_resolution_clock::now();
				auto duration_BVI = duration_cast<microseconds>(stop_BVI - start_BVI);
				
				stringstream_BVI << to_string(S) << " " << duration_BVI.count() << endl;

				//VIAE
				A_type A4 = copy_A(A);
				auto start_VIAE = high_resolution_clock::now();

				V_type V_AE_approx_solution_tuple = value_iteration_action_elimination(S, R, A4, P, gamma, epsilon);
				vector<double> V_AE_approx_solution = get<0>(V_AE_approx_solution_tuple);

				auto stop_VIAE = high_resolution_clock::now();
				auto duration_VIAE = duration_cast<microseconds>(stop_VIAE - start_VIAE);

				stringstream_VIAE << to_string(S) << " " << duration_VIAE.count() << endl;

				//VIAEH
				A_type A5 = copy_A(A);
				auto start_VIAEH = high_resolution_clock::now();

				V_type V_AE_H_approx_solution_tuple = value_iteration_action_elimination_heaps(S, R, A5, P, gamma, epsilon);
				vector<double> V_AE_H_approx_solution = get<0>(V_AE_H_approx_solution_tuple);

				auto stop_VIAEH = high_resolution_clock::now();
				auto duration_VIAEH = duration_cast<microseconds>(stop_VIAEH - start_VIAEH);

				stringstream_VIAEH << to_string(S) << " " << duration_VIAEH.count() << endl;

				//VIAEHL
				A_type A8 = copy_A(A);
				auto start_VIAEHL = high_resolution_clock::now();

				V_type VIAEHL_approx_solution_tuple = value_iteration_action_elimination_heaps_lower_bound_approx(S, R, A8, P, gamma, epsilon);
				vector<double> VIAEHL_approx_solution = get<0>(VIAEHL_approx_solution_tuple);

				auto stop_VIAEHL = high_resolution_clock::now();
				auto duration_VIAEHL = duration_cast<microseconds>(stop_VIAEHL - start_VIAEHL);

				stringstream_VIAEHL << to_string(S) << " " << duration_VIAEHL.count() << endl;
				
				//BAO
				A_type A9 = copy_A(A);
				auto start_BAO = high_resolution_clock::now();

				V_type BAO_approx_solution_tuple = value_iteration_BAO(S, R, A9, P, gamma, epsilon);
				vector<double> BAO_approx_solution = get<0>(BAO_approx_solution_tuple);

				auto stop_BAO = high_resolution_clock::now();
				auto duration_BAO = duration_cast<microseconds>(stop_BAO - start_BAO);

				stringstream_BAO << to_string(S) << " " << duration_BAO.count() << endl;
				
				//They should in theory all be epsilon from true value, and therefore, at most be 2 * epsilon from each other
				if (abs_max_diff_vectors(V_approx_solution, V_heap_approx) > (2 * epsilon)){
						printf("DIFFERENCE\n");
				}
				if (abs_max_diff_vectors(V_AE_approx_solution, V_AE_H_approx_solution) > (2 * epsilon)){
						printf("DIFFERENCE\n");
				}
				if (abs_max_diff_vectors(V_bounded_approx_solution, V_approx_solution_upper) > (2 * epsilon)){
						printf("DIFFERENCE\n");
				}
				if (abs_max_diff_vectors(VIAEHL_approx_solution, BAO_approx_solution) > (2 * epsilon)){
						printf("DIFFERENCE\n");
				}

		}

		//WRITE ALL DATA TO THEIR RESPECTVIE FILES	
		write_stringstream_to_file(stringstream_VI, output_stream_VI, file_name_VI);
		write_stringstream_to_file(stringstream_VIU, output_stream_VIU, file_name_VIU);
		write_stringstream_to_file(stringstream_BVI, output_stream_BVI, file_name_BVI);
		write_stringstream_to_file(stringstream_VIH, output_stream_VIH, file_name_VIH);
		write_stringstream_to_file(stringstream_VIAE, output_stream_VIAE, file_name_VIAE);
		write_stringstream_to_file(stringstream_VIAEH, output_stream_VIAEH, file_name_VIAEH);
		write_stringstream_to_file(stringstream_VIAEHL, output_stream_VIAEHL, file_name_VIAEHL);
		write_stringstream_to_file(stringstream_BAO, output_stream_BAO, file_name_BAO);
}

//VARYING NUMBER OF BOTH STATES AND ACTIONS
void write_meta_data_to_dat_file_number_of_states_and_actions(ostringstream& string_stream, double epsilon, double gamma, double upper_reward, double non_zero_transition, double action_prob, int A_S_starting_value, int A_S_finishing_value, int A_S_increment){
		time_t time_now = time(0);
		string_stream << "# META DATA" << endl;
		string_stream << "# " << endl;
		string_stream << "# " << "experiment run at: " << ctime(&time_now);
		string_stream << "# " << endl;
		string_stream << "# " << "gamma = " << gamma << endl;
		string_stream << "# " << "epsilon = " << epsilon << endl;
		string_stream << "# " << "non_zero_transition = " << non_zero_transition << endl;
		string_stream << "# " << "upper_reward = " << upper_reward << endl;
		string_stream << "# " << "action_prob = " << action_prob << endl;
		string_stream << "# " << endl;
		string_stream << "# VARIABLE IN THIS EXPERIMENT" << endl;
		string_stream << "# " << "Number of actions and states, A_S, is varied from " << A_S_starting_value << " to " << A_S_finishing_value << " with " << A_S_increment << " increment" << endl;
		string_stream << "# " << endl;
		string_stream << "# ACTUAL DATA" << endl;
		string_stream << "# number of actions and states | microseconds" << endl;
}

void create_data_tables_number_of_states_and_actions(string filename, int A_S_max, double epsilon, double gamma, double upper_reward, double non_zero_transition){

		//the stringstreams to create the test for the files
		ostringstream stringstream_BVI;
		ostringstream stringstream_VI;
		ostringstream stringstream_VIU;
		ostringstream stringstream_VIH;
		ostringstream stringstream_VIAE;
		ostringstream stringstream_VIAEH;

		//the file output objects
		ofstream output_stream_BVI;
		ofstream output_stream_VI;
		ofstream output_stream_VIU;
		ofstream output_stream_VIH;
		ofstream output_stream_VIAE;
		ofstream output_stream_VIAEH;
		
		//set the name of the file to write to
		string file_name_BVI = "data_tables/number_of_states_and_actions/" + filename + "_BVI.dat"; 
		string file_name_VI =  "data_tables/number_of_states_and_actions/" + filename + "_VI.dat";
		string file_name_VIU =  "data_tables/number_of_states_and_actions/" + filename + "_VIU.dat";
		string file_name_VIH = "data_tables/number_of_states_and_actions/" + filename + "_VIH.dat";
		string file_name_VIAE = "data_tables/number_of_states_and_actions/" + filename + "_VIAE.dat";
		string file_name_VIAEH = "data_tables/number_of_states_and_actions/" + filename + "_VIAEH.dat";

		//The varying parameters
		int A_S_starting_value = 50;
		int A_S_finishing_value = A_S_max;
		int A_S_increment = 50;

		//hardcoded parameter
		double action_prob = 1.0;

		write_meta_data_to_dat_file_number_of_states_and_actions(stringstream_VI, epsilon, gamma, upper_reward, non_zero_transition, action_prob, A_S_starting_value, A_S_finishing_value, A_S_increment);
		write_meta_data_to_dat_file_number_of_states_and_actions(stringstream_VIU, epsilon, gamma, upper_reward, non_zero_transition, action_prob, A_S_starting_value, A_S_finishing_value, A_S_increment);
		write_meta_data_to_dat_file_number_of_states_and_actions(stringstream_VIH, epsilon, gamma, upper_reward, non_zero_transition, action_prob, A_S_starting_value, A_S_finishing_value, A_S_increment);
		write_meta_data_to_dat_file_number_of_states_and_actions(stringstream_BVI, epsilon, gamma, upper_reward, non_zero_transition, action_prob, A_S_starting_value, A_S_finishing_value, A_S_increment);
		write_meta_data_to_dat_file_number_of_states_and_actions(stringstream_VIAE, epsilon, gamma, upper_reward, non_zero_transition, action_prob, A_S_starting_value, A_S_finishing_value, A_S_increment);
		write_meta_data_to_dat_file_number_of_states_and_actions(stringstream_VIAEH, epsilon, gamma, upper_reward, non_zero_transition, action_prob, A_S_starting_value, A_S_finishing_value, A_S_increment);

		for (int A_S = A_S_starting_value; A_S <= A_S_finishing_value; A_S = A_S + A_S_increment){
				
				printf("Beginning iteration A_S = %d\n", A_S);

				//GENERATE THE MDP
				int seed = time(0);
				auto MDP = generate_random_MDP_with_variable_parameters(A_S, A_S, action_prob, non_zero_transition, upper_reward, seed);
				R_type R = get<0>(MDP);
				A_type A = get<1>(MDP);
				P_type P = get<2>(MDP);

				//VI testing
				//TODO: make method that takes arguments and performs the writing of data, takes filename, MDP etc. as argument
				A_type A1 = copy_A(A);
				auto start_VI = high_resolution_clock::now();

				V_type V_approx_solution_tuple = value_iteration(A_S, R, A1, P, gamma, epsilon);
                vector<double> V_approx_solution = get<0>(V_approx_solution_tuple);

				auto stop_VI = high_resolution_clock::now();
				auto duration_VI = duration_cast<microseconds>(stop_VI - start_VI);

				stringstream_VI << to_string(A_S) << " " << duration_VI.count() << endl;

				//VIU testing
				A_type A6 = copy_A(A);
				auto start_VIU = high_resolution_clock::now();

				V_type V_approx_solution_upper_tuple = value_iteration_upper(A_S, R, A6, P, gamma, epsilon);
				vector<double> V_approx_solution_upper = get<0>(V_approx_solution_upper_tuple);

				auto stop_VIU = high_resolution_clock::now();
				auto duration_VIU = duration_cast<microseconds>(stop_VIU - start_VIU);

				stringstream_VIU << to_string(A_S) << " " << duration_VIU.count() << endl;

				//VIH testing
				A_type A2 = copy_A(A);
				auto start_VIH = high_resolution_clock::now();

				V_type V_heap_approx_tuple = value_iteration_with_heap(A_S, R, A2, P, gamma, epsilon);
				vector<double> V_heap_approx = get<0>(V_heap_approx_tuple);

				auto stop_VIH = high_resolution_clock::now();
				auto duration_VIH = duration_cast<microseconds>(stop_VIH - start_VIH);

				stringstream_VIH << to_string(A_S) << " " << duration_VIH.count() << endl;

				//BVI
				A_type A3 = copy_A(A);
				auto start_BVI = high_resolution_clock::now();

				V_type V_bounded_approx_solution_tuple = bounded_value_iteration(A_S, R, A3, P, gamma, epsilon);
				vector<double> V_bounded_approx_solution = get<0>(V_bounded_approx_solution_tuple);

				auto stop_BVI = high_resolution_clock::now();
				auto duration_BVI = duration_cast<microseconds>(stop_BVI - start_BVI);
				
				stringstream_BVI << to_string(A_S) << " " << duration_BVI.count() << endl;

				//VIAE
				A_type A4 = copy_A(A);
				auto start_VIAE = high_resolution_clock::now();

				V_type V_AE_approx_solution_tuple = value_iteration_action_elimination(A_S, R, A4, P, gamma, epsilon);
				vector<double> V_AE_approx_solution = get<0>(V_AE_approx_solution_tuple);

				auto stop_VIAE = high_resolution_clock::now();
				auto duration_VIAE = duration_cast<microseconds>(stop_VIAE - start_VIAE);

				stringstream_VIAE << to_string(A_S) << " " << duration_VIAE.count() << endl;

				//VIAEH
				A_type A5 = copy_A(A);
				auto start_VIAEH = high_resolution_clock::now();

				V_type V_AE_H_approx_solution_tuple = value_iteration_action_elimination_heaps(A_S, R, A5, P, gamma, epsilon);
				vector<double> V_AE_H_approx_solution = get<0>(V_AE_H_approx_solution_tuple);

				auto stop_VIAEH = high_resolution_clock::now();
				auto duration_VIAEH = duration_cast<microseconds>(stop_VIAEH - start_VIAEH);

				stringstream_VIAEH << to_string(A_S) << " " << duration_VIAEH.count() << endl;

				//They should in theory all be epsilon from true value, and therefore, at most be 2 * epsilon from each other
				if (abs_max_diff_vectors(V_approx_solution, V_heap_approx) > (2 * epsilon)){
						printf("DIFFERENCE\n");
				}
				if (abs_max_diff_vectors(V_AE_approx_solution, V_AE_H_approx_solution) > (2 * epsilon)){
						printf("DIFFERENCE\n");
				}
				if (abs_max_diff_vectors(V_bounded_approx_solution, V_approx_solution_upper) > (2 * epsilon)){
						printf("DIFFERENCE\n");
				}

		}

		//WRITE ALL DATA TO THEIR RESPECTVIE FILES	
		write_stringstream_to_file(stringstream_VI, output_stream_VI, file_name_VI);
		write_stringstream_to_file(stringstream_VIU, output_stream_VIU, file_name_VIU);
		write_stringstream_to_file(stringstream_BVI, output_stream_BVI, file_name_BVI);
		write_stringstream_to_file(stringstream_VIH, output_stream_VIH, file_name_VIH);
		write_stringstream_to_file(stringstream_VIAE, output_stream_VIAE, file_name_VIAE);
		write_stringstream_to_file(stringstream_VIAEH, output_stream_VIAEH, file_name_VIAEH);
}

//VARYING NUMBER OF ACTIONS
void write_meta_data_to_dat_file_number_of_actions(ostringstream& string_stream, int S, double epsilon, double gamma, double upper_reward, double non_zero_transition, double action_prob, int A_starting_value, int A_finishing_value, int A_increment){
		time_t time_now = time(0);
		string_stream << "# META DATA" << endl;
		string_stream << "# " << endl;
		string_stream << "# " << "experiment run at: " << ctime(&time_now);
		string_stream << "# " << endl;
		string_stream << "# " << "gamma = " << gamma << endl;
		string_stream << "# " << "epsilon = " << epsilon << endl;
		string_stream << "# " << "S = " << S << endl;
		string_stream << "# " << "non_zero_transition = " << non_zero_transition << endl;
		string_stream << "# " << "upper_reward = " << upper_reward << endl;
		string_stream << "# " << "action_prob = " << action_prob << endl;
		string_stream << "# " << endl;
		string_stream << "# VARIABLE IN THIS EXPERIMENT" << endl;
		string_stream << "# " << "Number of actions, A, is varied from " << A_starting_value << " to " << A_finishing_value << " with " << A_increment << " increment" << endl;
		string_stream << "# " << endl;
		string_stream << "# ACTUAL DATA" << endl;
		string_stream << "# number of actions | microseconds" << endl;
}

void create_data_tables_number_of_actions(string filename, int S, int A_max, double epsilon, double gamma, double upper_reward, double non_zero_transition){

		//the stringstreams to create the test for the files
		ostringstream stringstream_BVI;
		ostringstream stringstream_VI;
		ostringstream stringstream_VIU;
		ostringstream stringstream_VIH;
		ostringstream stringstream_VIAE;
		ostringstream stringstream_VIAEH;
		ostringstream stringstream_VIAEHL;
		ostringstream stringstream_BAO;

		//the file output objects
		ofstream output_stream_BVI;
		ofstream output_stream_VI;
		ofstream output_stream_VIU;
		ofstream output_stream_VIH;
		ofstream output_stream_VIAE;
		ofstream output_stream_VIAEH;
		ofstream output_stream_VIAEHL;
		ofstream output_stream_BAO;
		
		//set the name of the file to write to
		string file_name_BVI = "data_tables/number_of_actions/" + filename + "_BVI.dat"; 
		string file_name_VI =  "data_tables/number_of_actions/" + filename + "_VI.dat";
		string file_name_VIU =  "data_tables/number_of_actions/" + filename + "_VIU.dat";
		string file_name_VIH = "data_tables/number_of_actions/" + filename + "_VIH.dat";
		string file_name_VIAE = "data_tables/number_of_actions/" + filename + "_VIAE.dat";
		string file_name_VIAEH = "data_tables/number_of_actions/" + filename + "_VIAEH.dat";
		string file_name_VIAEHL = "data_tables/number_of_actions/" + filename + "_VIAEHL.dat";
		string file_name_BAO = "data_tables/number_of_actions/" + filename + "_BAO.dat";

		//The varying parameters
		int A_starting_value = 50;
		int A_finishing_value = A_max;
		int A_increment = 50;

		//hardcoded parameter
		double action_prob = 1.0;

		//write meta data to all stringstreams as first in their respective files
		write_meta_data_to_dat_file_number_of_actions(stringstream_VI, S, epsilon, gamma, upper_reward, non_zero_transition, action_prob, A_starting_value, A_finishing_value, A_increment);
		write_meta_data_to_dat_file_number_of_actions(stringstream_VIU, S, epsilon, gamma, upper_reward, non_zero_transition, action_prob, A_starting_value, A_finishing_value, A_increment);
		write_meta_data_to_dat_file_number_of_actions(stringstream_VIH, S, epsilon, gamma, upper_reward, non_zero_transition, action_prob, A_starting_value, A_finishing_value, A_increment);
		write_meta_data_to_dat_file_number_of_actions(stringstream_BVI, S, epsilon, gamma, upper_reward, non_zero_transition, action_prob, A_starting_value, A_finishing_value, A_increment);
		write_meta_data_to_dat_file_number_of_actions(stringstream_VIAE, S, epsilon, gamma, upper_reward, non_zero_transition, action_prob, A_starting_value, A_finishing_value, A_increment);
		write_meta_data_to_dat_file_number_of_actions(stringstream_VIAEH, S, epsilon, gamma, upper_reward, non_zero_transition, action_prob, A_starting_value, A_finishing_value, A_increment);

		for (int A_num = A_starting_value; A_num <= A_finishing_value; A_num = A_num + A_increment){
				
				printf("Beginning iteration A_num = %d\n", A_num);

				//auto MDP = generate_random_MDP_exponential_distributed_rewards(S, A_num, action_prob, non_zero_transition, 0.02, seed);
				//auto MDP = generate_random_MDP_with_variable_parameters(S, A_num, action_prob, non_zero_transition, 1000, seed);
				//GENERATE THE MDP
				int seed = time(0);
				auto MDP = generate_random_MDP_normal_distributed_rewards(S, A_num, 1.0, S, seed, 1000, 10);
				//auto MDP = generate_random_MDP_exponential_distributed_rewards(S, A_num, 1.0, non_zero_transition, 0.02, seed);
				//auto MDP = generate_random_MDP_with_variable_parameters(S, A_num, 1.0, non_zero_transition, upper_reward, seed);
				R_type R = get<0>(MDP);
				A_type A = get<1>(MDP);
				P_type P = get<2>(MDP);

				//VI testing
				//TODO: make method that takes arguments and performs the writing of data, takes filename, MDP etc. as argument
				A_type A1 = copy_A(A);
				auto start_VI = high_resolution_clock::now();

				V_type V_approx_solution_tuple = value_iteration(S, R, A1, P, gamma, epsilon);
                vector<double> V_approx_solution = get<0>(V_approx_solution_tuple);

				auto stop_VI = high_resolution_clock::now();
				auto duration_VI = duration_cast<microseconds>(stop_VI - start_VI);

				stringstream_VI << to_string(A_num) << " " << duration_VI.count() << endl;

				//VIU testing
				A_type A6 = copy_A(A);
				auto start_VIU = high_resolution_clock::now();

				V_type V_approx_solution_upper_tuple = value_iteration_upper(S, R, A6, P, gamma, epsilon);
				vector<double> V_approx_solution_upper = get<0>(V_approx_solution_upper_tuple);

				auto stop_VIU = high_resolution_clock::now();
				auto duration_VIU = duration_cast<microseconds>(stop_VIU - start_VIU);

				stringstream_VIU << to_string(A_num) << " " << duration_VIU.count() << endl;

				//VIH testing
				A_type A2 = copy_A(A);
				auto start_VIH = high_resolution_clock::now();

				V_type V_heap_approx_tuple = value_iteration_with_heap(S, R, A2, P, gamma, epsilon);
				vector<double> V_heap_approx = get<0>(V_heap_approx_tuple);

				auto stop_VIH = high_resolution_clock::now();
				auto duration_VIH = duration_cast<microseconds>(stop_VIH - start_VIH);

				stringstream_VIH << to_string(A_num) << " " << duration_VIH.count() << endl;

				//BVI
				A_type A3 = copy_A(A);
				auto start_BVI = high_resolution_clock::now();

				V_type V_bounded_approx_solution_tuple = bounded_value_iteration(S, R, A3, P, gamma, epsilon);
				vector<double> V_bounded_approx_solution = get<0>(V_bounded_approx_solution_tuple);

				auto stop_BVI = high_resolution_clock::now();
				auto duration_BVI = duration_cast<microseconds>(stop_BVI - start_BVI);
				
				stringstream_BVI << to_string(A_num) << " " << duration_BVI.count() << endl;

				//VIAE
				A_type A4 = copy_A(A);
				auto start_VIAE = high_resolution_clock::now();

				V_type V_AE_approx_solution_tuple = value_iteration_action_elimination(S, R, A4, P, gamma, epsilon);
				vector<double> V_AE_approx_solution = get<0>(V_AE_approx_solution_tuple);

				auto stop_VIAE = high_resolution_clock::now();
				auto duration_VIAE = duration_cast<microseconds>(stop_VIAE - start_VIAE);

				stringstream_VIAE << to_string(A_num) << " " << duration_VIAE.count() << endl;

				//VIAEH
				A_type A5 = copy_A(A);
				auto start_VIAEH = high_resolution_clock::now();

				V_type V_AE_H_approx_solution_tuple = value_iteration_action_elimination_heaps(S, R, A5, P, gamma, epsilon);
				vector<double> V_AE_H_approx_solution = get<0>(V_AE_H_approx_solution_tuple);

				auto stop_VIAEH = high_resolution_clock::now();
				auto duration_VIAEH = duration_cast<microseconds>(stop_VIAEH - start_VIAEH);

				stringstream_VIAEH << to_string(A_num) << " " << duration_VIAEH.count() << endl;

				//VIAEHL
				A_type A9 = copy_A(A);
				auto start_VIAEHL = high_resolution_clock::now();

				V_type VIAEHL_approx_solution_tuple = value_iteration_action_elimination_heaps_lower_bound_approx(S, R, A9, P, gamma, epsilon);
				vector<double> VIAEHL_approx_solution = get<0>(VIAEHL_approx_solution_tuple);

				auto stop_VIAEHL = high_resolution_clock::now();
				auto duration_VIAEHL = duration_cast<microseconds>(stop_VIAEHL - start_VIAEHL);

				stringstream_VIAEHL << to_string(A_num) << " " << duration_VIAEHL.count() << endl;
				
				//BAO
				A_type A8 = copy_A(A);
				auto start_BAO = high_resolution_clock::now();

				V_type BAO_approx_solution_tuple = value_iteration_BAO(S, R, A8, P, gamma, epsilon);
				vector<double> BAO_approx_solution = get<0>(BAO_approx_solution_tuple);

				auto stop_BAO = high_resolution_clock::now();
				auto duration_BAO = duration_cast<microseconds>(stop_BAO - start_BAO);

				stringstream_BAO << to_string(A_num) << " " << duration_BAO.count() << endl;
			
				//They should in theory all be epsilon from true value, and therefore, at most be 2 * epsilon from each other
				if (abs_max_diff_vectors(V_approx_solution, V_heap_approx) > (2 * epsilon)){
						printf("DIFFERENCE\n");
				}
				if (abs_max_diff_vectors(V_AE_approx_solution, V_AE_H_approx_solution) > (2 * epsilon)){
						printf("DIFFERENCE\n");
				}
				if (abs_max_diff_vectors(V_bounded_approx_solution, V_approx_solution_upper) > (2 * epsilon)){
						printf("DIFFERENCE\n");
				}
				if (abs_max_diff_vectors(VIAEHL_approx_solution	, BAO_approx_solution) > (2 * epsilon)){
						printf("DIFFERENCE\n");
				}

		}

		//WRITE ALL DATA TO THEIR RESPECTVIE FILES	
		write_stringstream_to_file(stringstream_VI, output_stream_VI, file_name_VI);
		write_stringstream_to_file(stringstream_VIU, output_stream_VIU, file_name_VIU);
		write_stringstream_to_file(stringstream_BVI, output_stream_BVI, file_name_BVI);
		write_stringstream_to_file(stringstream_VIH, output_stream_VIH, file_name_VIH);
		write_stringstream_to_file(stringstream_VIAE, output_stream_VIAE, file_name_VIAE);
		write_stringstream_to_file(stringstream_VIAEH, output_stream_VIAEH, file_name_VIAEH);
		write_stringstream_to_file(stringstream_VIAEHL, output_stream_VIAEHL, file_name_VIAEHL);
		write_stringstream_to_file(stringstream_BAO, output_stream_BAO, file_name_BAO);
}

//VARYING REWARD PROB
void write_meta_data_to_dat_file_reward_prob(ostringstream& string_stream, int S, int A_num, double epsilon, double gamma, double reward_factor, double upper_reward, double non_zero_transition, double action_prob, double reward_prob_starting_value, double reward_prob_finishing_value, double reward_prob_increment){
		time_t time_now = time(0);
		string_stream << "# META DATA" << endl;
		string_stream << "# " << endl;
		string_stream << "# " << "experiment run at: " << ctime(&time_now);
		string_stream << "# " << endl;
		string_stream << "# " << "gamma = " << gamma << endl;
		string_stream << "# " << "epsilon = " << epsilon << endl;
		string_stream << "# " << "S = " << S << endl;
		string_stream << "# " << "A_num = " << A_num << endl;
		string_stream << "# " << "non_zero_transition = " << non_zero_transition << endl;
		string_stream << "# " << "reward_factor = " << reward_factor << endl;
		string_stream << "# " << "upper_reward = " << upper_reward << endl;
		string_stream << "# " << "action_prob = " << action_prob << endl;
		string_stream << "# " << endl;
		string_stream << "# VARIABLE IN THIS EXPERIMENT" << endl;
		string_stream << "# " << "The probability that a reward is multiplied by reward_factor after initial sampling, reward_prob, is varied from " << reward_prob_starting_value << " to " << reward_prob_finishing_value << " with " << reward_prob_increment << " increment" << endl;
		string_stream << "# " << endl;
		string_stream << "# ACTUAL DATA" << endl;
		string_stream << "# reward_prob | microseconds" << endl;
}

void create_data_tables_rewards(string filename, int S, int A_num, double epsilon, double gamma, double reward_factor, double reward_prob, double upper_reward, double action_prob, double non_zero_transition){

		//the stringstreams to create the test for the files
		ostringstream stringstream_BVI;
		ostringstream stringstream_VI;
		ostringstream stringstream_VIU;
		ostringstream stringstream_VIH;
		ostringstream stringstream_VIAE;
		ostringstream stringstream_VIAEH;

		//the file output objects
		ofstream output_stream_BVI;
		ofstream output_stream_VI;
		ofstream output_stream_VIU;
		ofstream output_stream_VIH;
		ofstream output_stream_VIAE;
		ofstream output_stream_VIAEH;
		
		//set the name of the file to write to
		string file_name_BVI = "data_tables/reward_dist/" + filename + "_BVI.dat"; 
		string file_name_VI =  "data_tables/reward_dist/" + filename + "_VI.dat";
		string file_name_VIU =  "data_tables/reward_dist/" + filename + "_VIU.dat";
		string file_name_VIH = "data_tables/reward_dist/" + filename + "_VIH.dat";
		string file_name_VIAE = "data_tables/reward_dist/" + filename + "_VIAE.dat";
		string file_name_VIAEH = "data_tables/reward_dist/" + filename + "_VIAEH.dat";

		//The varying parameters
		double reward_prob_starting_value = 0.0;
		double reward_prob_finishing_value = 1.0;
		double reward_prob_increment = 0.01;

		write_meta_data_to_dat_file_reward_prob(stringstream_VI, S, A_num, epsilon, gamma, reward_factor, upper_reward, non_zero_transition, action_prob, reward_prob_starting_value, reward_prob_finishing_value, reward_prob_increment);
		write_meta_data_to_dat_file_reward_prob(stringstream_VIU, S, A_num, epsilon, gamma, reward_factor, upper_reward, non_zero_transition, action_prob, reward_prob_starting_value, reward_prob_finishing_value, reward_prob_increment);
		write_meta_data_to_dat_file_reward_prob(stringstream_VIH, S, A_num, epsilon, gamma, reward_factor, upper_reward, non_zero_transition, action_prob, reward_prob_starting_value, reward_prob_finishing_value, reward_prob_increment);
		write_meta_data_to_dat_file_reward_prob(stringstream_BVI, S, A_num, epsilon, gamma, reward_factor, upper_reward, non_zero_transition, action_prob, reward_prob_starting_value, reward_prob_finishing_value, reward_prob_increment);
		write_meta_data_to_dat_file_reward_prob(stringstream_VIAE, S, A_num, epsilon, gamma, reward_factor, upper_reward, non_zero_transition, action_prob, reward_prob_starting_value, reward_prob_finishing_value, reward_prob_increment);
		write_meta_data_to_dat_file_reward_prob(stringstream_VIAEH, S, A_num, epsilon, gamma, reward_factor, upper_reward, non_zero_transition, action_prob, reward_prob_starting_value, reward_prob_finishing_value, reward_prob_increment);

		for (double reward_prob = reward_prob_starting_value; reward_prob <= reward_prob_finishing_value; reward_prob = reward_prob + reward_prob_increment){
				
				printf("Beginning iteration reward_prob = %f\n", reward_prob);

				//GENERATE THE MDP
				int seed = time(0);
				auto MDP = generate_random_MDP_with_variable_parameters_and_reward(S, A_num, action_prob, non_zero_transition, reward_factor, reward_prob, upper_reward, seed);
				R_type R = get<0>(MDP);
				A_type A = get<1>(MDP);
				P_type P = get<2>(MDP);

				//VI testing
				//TODO: make method that takes arguments and performs the writing of data, takes filename, MDP etc. as argument
				A_type A1 = copy_A(A);
				auto start_VI = high_resolution_clock::now();

				V_type V_approx_solution_tuple = value_iteration(S, R, A1, P, gamma, epsilon);
                vector<double> V_approx_solution = get<0>(V_approx_solution_tuple);

				auto stop_VI = high_resolution_clock::now();
				auto duration_VI = duration_cast<microseconds>(stop_VI - start_VI);

				stringstream_VI << to_string(reward_prob) << " " << duration_VI.count() << endl;

				//VIU testing
				A_type A6 = copy_A(A);
				auto start_VIU = high_resolution_clock::now();

				V_type V_approx_solution_upper_tuple = value_iteration_upper(S, R, A6, P, gamma, epsilon);
				vector<double> V_approx_solution_upper = get<0>(V_approx_solution_upper_tuple);

				auto stop_VIU = high_resolution_clock::now();
				auto duration_VIU = duration_cast<microseconds>(stop_VIU - start_VIU);

				stringstream_VIU << to_string(reward_prob) << " " << duration_VIU.count() << endl;

				//VIH testing
				A_type A2 = copy_A(A);
				auto start_VIH = high_resolution_clock::now();

				V_type V_heap_approx_tuple = value_iteration_with_heap(S, R, A2, P, gamma, epsilon);
				vector<double> V_heap_approx = get<0>(V_heap_approx_tuple);

				auto stop_VIH = high_resolution_clock::now();
				auto duration_VIH = duration_cast<microseconds>(stop_VIH - start_VIH);

				stringstream_VIH << to_string(reward_prob) << " " << duration_VIH.count() << endl;

				//BVI
				A_type A3 = copy_A(A);
				auto start_BVI = high_resolution_clock::now();

				V_type V_bounded_approx_solution_tuple = bounded_value_iteration(S, R, A3, P, gamma, epsilon);
				vector<double> V_bounded_approx_solution = get<0>(V_bounded_approx_solution_tuple);

				auto stop_BVI = high_resolution_clock::now();
				auto duration_BVI = duration_cast<microseconds>(stop_BVI - start_BVI);
				
				stringstream_BVI << to_string(reward_prob) << " " << duration_BVI.count() << endl;

				//VIAE
				A_type A4 = copy_A(A);
				auto start_VIAE = high_resolution_clock::now();

				V_type V_AE_approx_solution_tuple = value_iteration_action_elimination(S, R, A4, P, gamma, epsilon);
				vector<double> V_AE_approx_solution = get<0>(V_AE_approx_solution_tuple);

				auto stop_VIAE = high_resolution_clock::now();
				auto duration_VIAE = duration_cast<microseconds>(stop_VIAE - start_VIAE);

				stringstream_VIAE << to_string(reward_prob) << " " << duration_VIAE.count() << endl;

				//VIAEH
				A_type A5 = copy_A(A);
				auto start_VIAEH = high_resolution_clock::now();

				V_type V_AE_H_approx_solution_tuple = value_iteration_action_elimination_heaps(S, R, A5, P, gamma, epsilon);
				vector<double> V_AE_H_approx_solution = get<0>(V_AE_H_approx_solution_tuple);

				auto stop_VIAEH = high_resolution_clock::now();
				auto duration_VIAEH = duration_cast<microseconds>(stop_VIAEH - start_VIAEH);

				stringstream_VIAEH << to_string(reward_prob) << " " << duration_VIAEH.count() << endl;

				//They should in theory all be epsilon from true value, and therefore, at most be 2 * epsilon from each other
				if (abs_max_diff_vectors(V_approx_solution, V_heap_approx) > (2 * epsilon)){
						printf("DIFFERENCE\n");
				}
				if (abs_max_diff_vectors(V_AE_approx_solution, V_AE_H_approx_solution) > (2 * epsilon)){
						printf("DIFFERENCE\n");
				}
				if (abs_max_diff_vectors(V_bounded_approx_solution, V_approx_solution_upper) > (2 * epsilon)){
						printf("DIFFERENCE\n");
				}

		}

		//WRITE ALL DATA TO THEIR RESPECTVIE FILES	
		write_stringstream_to_file(stringstream_VI, output_stream_VI, file_name_VI);
		write_stringstream_to_file(stringstream_VIU, output_stream_VIU, file_name_VIU);
		write_stringstream_to_file(stringstream_BVI, output_stream_BVI, file_name_BVI);
		write_stringstream_to_file(stringstream_VIH, output_stream_VIH, file_name_VIH);
		write_stringstream_to_file(stringstream_VIAE, output_stream_VIAE, file_name_VIAE);
		write_stringstream_to_file(stringstream_VIAEH, output_stream_VIAEH, file_name_VIAEH);
}

//VARYING TRANSITION PROB
void write_meta_data_to_dat_file_transition_prob(ostringstream& string_stream, int S, int A_num, double epsilon, double gamma, double upper_reward, double action_prob, double transition_prob_starting_value, double transition_prob_finishing_value, double transition_prob_increment){
		time_t time_now = time(0);
		string_stream << "# META DATA" << endl;
		string_stream << "# " << endl;
		string_stream << "# " << "experiment run at: " << ctime(&time_now);
		string_stream << "# " << endl;
		string_stream << "# " << "gamma = " << gamma << endl;
		string_stream << "# " << "epsilon = " << epsilon << endl;
		string_stream << "# " << "S = " << S << endl;
		string_stream << "# " << "A_num = " << A_num << endl;
		string_stream << "# " << "upper_reward = " << upper_reward << endl;
		string_stream << "# " << "action_prob = " << action_prob << endl;
		string_stream << "# " << endl;
		string_stream << "# VARIABLE IN THIS EXPERIMENT" << endl;
		string_stream << "# " << "The probability that a state has a non-zero transition probability, transition_prob, is varied from " << transition_prob_starting_value << " to " << transition_prob_finishing_value << " with " << transition_prob_increment << " increment" << endl;
		string_stream << "# " << endl;
		string_stream << "# ACTUAL DATA" << endl;
		string_stream << "# transition_prob | microseconds" << endl;
}

void create_data_tables_transition_prob(string filename, int S, int A_num, double epsilon, double gamma, double upper_reward){

		//the stringstreams to create the test for the files
		ostringstream stringstream_BVI;
		ostringstream stringstream_VI;
		ostringstream stringstream_VIU;
		ostringstream stringstream_VIH;
		ostringstream stringstream_VIAE;
		ostringstream stringstream_VIAEH;

		//the file output objects
		ofstream output_stream_BVI;
		ofstream output_stream_VI;
		ofstream output_stream_VIU;
		ofstream output_stream_VIH;
		ofstream output_stream_VIAE;
		ofstream output_stream_VIAEH;
		
		//set the name of the file to write to
		string file_name_BVI = "data_tables/transition_prob/" + filename + "_BVI.dat"; 
		string file_name_VI =  "data_tables/transition_prob/" + filename + "_VI.dat";
		string file_name_VIU =  "data_tables/transition_prob/" + filename + "_VIU.dat";
		string file_name_VIH = "data_tables/transition_prob/" + filename + "_VIH.dat";
		string file_name_VIAE = "data_tables/transition_prob/" + filename + "_VIAE.dat";
		string file_name_VIAEH = "data_tables/transition_prob/" + filename + "_VIAEH.dat";

		//The varying parameters
		double transition_prob_starting_value = 0.1;
		double transition_prob_finishing_value = 1.0;
		double transition_prob_increment = 0.05;
		
		//hardcoded parameter
		double action_prob = 1.0;
		
		write_meta_data_to_dat_file_transition_prob(stringstream_VI, S, A_num, epsilon, gamma, upper_reward, action_prob, transition_prob_starting_value, transition_prob_finishing_value, transition_prob_increment);
		write_meta_data_to_dat_file_transition_prob(stringstream_VIU, S, A_num, epsilon, gamma, upper_reward, action_prob, transition_prob_starting_value, transition_prob_finishing_value, transition_prob_increment);
		write_meta_data_to_dat_file_transition_prob(stringstream_VIH, S, A_num, epsilon, gamma, upper_reward, action_prob, transition_prob_starting_value, transition_prob_finishing_value, transition_prob_increment);
		write_meta_data_to_dat_file_transition_prob(stringstream_BVI, S, A_num, epsilon, gamma, upper_reward, action_prob, transition_prob_starting_value, transition_prob_finishing_value, transition_prob_increment);
		write_meta_data_to_dat_file_transition_prob(stringstream_VIAE, S, A_num, epsilon, gamma, upper_reward, action_prob, transition_prob_starting_value, transition_prob_finishing_value, transition_prob_increment);
		write_meta_data_to_dat_file_transition_prob(stringstream_VIAEH, S, A_num, epsilon, gamma, upper_reward, action_prob, transition_prob_starting_value, transition_prob_finishing_value, transition_prob_increment);

		for (double transition_prob = transition_prob_starting_value; transition_prob <= transition_prob_finishing_value; transition_prob = transition_prob + transition_prob_increment){
				
				printf("Beginning iteration transition_prob = %f\n", transition_prob);

				//GENERATE THE MDP
				int seed = time(0);
				auto MDP = generate_random_MDP_with_variable_parameters(S, A_num, action_prob, transition_prob, upper_reward, seed);
				R_type R = get<0>(MDP);
				A_type A = get<1>(MDP);
				P_type P = get<2>(MDP);

				//VI testing
				//TODO: make method that takes arguments and performs the writing of data, takes filename, MDP etc. as argument
				A_type A1 = copy_A(A);
				auto start_VI = high_resolution_clock::now();

				V_type V_approx_solution_tuple = value_iteration(S, R, A1, P, gamma, epsilon);
                vector<double> V_approx_solution = get<0>(V_approx_solution_tuple);

				auto stop_VI = high_resolution_clock::now();
				auto duration_VI = duration_cast<microseconds>(stop_VI - start_VI);

				stringstream_VI << to_string(transition_prob) << " " << duration_VI.count() << endl;

				//VIU testing
				A_type A6 = copy_A(A);
				auto start_VIU = high_resolution_clock::now();

				V_type V_approx_solution_upper_tuple = value_iteration_upper(S, R, A6, P, gamma, epsilon);
				vector<double> V_approx_solution_upper = get<0>(V_approx_solution_upper_tuple);

				auto stop_VIU = high_resolution_clock::now();
				auto duration_VIU = duration_cast<microseconds>(stop_VIU - start_VIU);

				stringstream_VIU << to_string(transition_prob) << " " << duration_VIU.count() << endl;

				//VIH testing
				A_type A2 = copy_A(A);
				auto start_VIH = high_resolution_clock::now();

				V_type V_heap_approx_tuple = value_iteration_with_heap(S, R, A2, P, gamma, epsilon);
				vector<double> V_heap_approx = get<0>(V_heap_approx_tuple);

				auto stop_VIH = high_resolution_clock::now();
				auto duration_VIH = duration_cast<microseconds>(stop_VIH - start_VIH);

				stringstream_VIH << to_string(transition_prob) << " " << duration_VIH.count() << endl;

				//BVI
				A_type A3 = copy_A(A);
				auto start_BVI = high_resolution_clock::now();

				V_type V_bounded_approx_solution_tuple = bounded_value_iteration(S, R, A3, P, gamma, epsilon);
				vector<double> V_bounded_approx_solution = get<0>(V_bounded_approx_solution_tuple);

				auto stop_BVI = high_resolution_clock::now();
				auto duration_BVI = duration_cast<microseconds>(stop_BVI - start_BVI);
				
				stringstream_BVI << to_string(transition_prob) << " " << duration_BVI.count() << endl;

				//VIAE
				A_type A4 = copy_A(A);
				auto start_VIAE = high_resolution_clock::now();

				V_type V_AE_approx_solution_tuple = value_iteration_action_elimination(S, R, A4, P, gamma, epsilon);
				vector<double> V_AE_approx_solution = get<0>(V_AE_approx_solution_tuple);

				auto stop_VIAE = high_resolution_clock::now();
				auto duration_VIAE = duration_cast<microseconds>(stop_VIAE - start_VIAE);

				stringstream_VIAE << to_string(transition_prob) << " " << duration_VIAE.count() << endl;

				//VIAEH
				A_type A5 = copy_A(A);
				auto start_VIAEH = high_resolution_clock::now();

				V_type V_AE_H_approx_solution_tuple = value_iteration_action_elimination_heaps(S, R, A5, P, gamma, epsilon);
				vector<double> V_AE_H_approx_solution = get<0>(V_AE_H_approx_solution_tuple);

				auto stop_VIAEH = high_resolution_clock::now();
				auto duration_VIAEH = duration_cast<microseconds>(stop_VIAEH - start_VIAEH);

				stringstream_VIAEH << to_string(transition_prob) << " " << duration_VIAEH.count() << endl;

				//They should in theory all be epsilon from true value, and therefore, at most be 2 * epsilon from each other
				if (abs_max_diff_vectors(V_approx_solution, V_heap_approx) > (2 * epsilon)){
						printf("DIFFERENCE\n");
				}
				if (abs_max_diff_vectors(V_AE_approx_solution, V_AE_H_approx_solution) > (2 * epsilon)){
						printf("DIFFERENCE\n");
				}
				if (abs_max_diff_vectors(V_bounded_approx_solution, V_approx_solution_upper) > (2 * epsilon)){
						printf("DIFFERENCE\n");
				}

		}

		//WRITE ALL DATA TO THEIR RESPECTVIE FILES	
		write_stringstream_to_file(stringstream_VI, output_stream_VI, file_name_VI);
		write_stringstream_to_file(stringstream_VIU, output_stream_VIU, file_name_VIU);
		write_stringstream_to_file(stringstream_BVI, output_stream_BVI, file_name_BVI);
		write_stringstream_to_file(stringstream_VIH, output_stream_VIH, file_name_VIH);
		write_stringstream_to_file(stringstream_VIAE, output_stream_VIAE, file_name_VIAE);
		write_stringstream_to_file(stringstream_VIAEH, output_stream_VIAEH, file_name_VIAEH);
}

//NUMBER OF STATES - ITERATIONS PLOT

void write_meta_data_to_dat_file_number_of_states_iterations(ostringstream& string_stream, int A_num, double epsilon, double gamma, double upper_reward, double non_zero_transition, double action_prob, int S_starting_value, int S_finishing_value, int S_increment){
		time_t time_now = time(0);
		string_stream << "# META DATA" << endl;
		string_stream << "# " << endl;
		string_stream << "# " << "experiment run at: " << ctime(&time_now);
		string_stream << "# " << endl;
		string_stream << "# " << "gamma = " << gamma << endl;
		string_stream << "# " << "epsilon = " << epsilon << endl;
		string_stream << "# " << "A = " << A_num << endl;
		string_stream << "# " << "non_zero_transition = " << non_zero_transition << endl;
		string_stream << "# " << "upper_reward = " << upper_reward << endl;
		string_stream << "# " << "action_prob = " << action_prob << endl;
		string_stream << "# " << endl;
		string_stream << "# VARIABLE IN THIS EXPERIMENT" << endl;
		string_stream << "# " << "Number of states, S, is varied from " << S_starting_value << " to " << S_finishing_value << " with " << S_increment << " increment" << endl;
		string_stream << "# " << endl;
		string_stream << "# ACTUAL DATA" << endl;
		string_stream << "# number of states | iterations" << endl;
}

void create_data_tables_number_of_states_iterations(string filename, int S_max, int A_num, double epsilon, double gamma, double upper_reward, double non_zero_transition){

		//the stringstreams to create the test for the files
		ostringstream stringstream_BVI;
		ostringstream stringstream_VI;
		ostringstream stringstream_VIU;
		ostringstream stringstream_VIH;
		ostringstream stringstream_VIAE;
		ostringstream stringstream_VIAEH;

		//the file output objects
		ofstream output_stream_BVI;
		ofstream output_stream_VI;
		ofstream output_stream_VIU;
		ofstream output_stream_VIH;
		ofstream output_stream_VIAE;
		ofstream output_stream_VIAEH;
		
		//set the name of the file to write to
		string file_name_BVI = "data_tables/number_of_states_iterations/" + filename + "_BVI.dat"; 
		string file_name_VI =  "data_tables/number_of_states_iterations/" + filename + "_VI.dat";
		string file_name_VIU =  "data_tables/number_of_states_iterations/" + filename + "_VIU.dat";
		string file_name_VIH = "data_tables/number_of_states_iterations/" + filename + "_VIH.dat";
		string file_name_VIAE = "data_tables/number_of_states_iterations/" + filename + "_VIAE.dat";
		string file_name_VIAEH = "data_tables/number_of_states_iterations/" + filename + "_VIAEH.dat";

		//The varying parameters
		int S_starting_value = 50;
		int S_finishing_value = S_max;
		int S_increment = 50;

		//hardcoded parameter
		double action_prob = 1.0;

		//write meta data to all stringstreams as first in their respective files
		
		write_meta_data_to_dat_file_number_of_states_iterations(stringstream_VI, A_num, epsilon, gamma, upper_reward, non_zero_transition, action_prob, S_starting_value, S_finishing_value, S_increment);
		write_meta_data_to_dat_file_number_of_states_iterations(stringstream_VIU, A_num, epsilon, gamma, upper_reward, non_zero_transition, action_prob, S_starting_value, S_finishing_value, S_increment);
		write_meta_data_to_dat_file_number_of_states_iterations(stringstream_BVI, A_num, epsilon, gamma, upper_reward, non_zero_transition, action_prob, S_starting_value, S_finishing_value, S_increment);
		write_meta_data_to_dat_file_number_of_states_iterations(stringstream_VIH, A_num, epsilon, gamma, upper_reward, non_zero_transition, action_prob, S_starting_value, S_finishing_value, S_increment);
		write_meta_data_to_dat_file_number_of_states_iterations(stringstream_VIAE, A_num, epsilon, gamma, upper_reward, non_zero_transition, action_prob, S_starting_value, S_finishing_value, S_increment);
		write_meta_data_to_dat_file_number_of_states_iterations(stringstream_VIAEH, A_num, epsilon, gamma, upper_reward, non_zero_transition, action_prob, S_starting_value, S_finishing_value, S_increment);

		for (int S = S_starting_value; S <= S_finishing_value; S = S + S_increment){
				
				printf("\nBeginning iteration S = %d\n", S);

				//GENERATE THE MDP
				int seed = time(0);
				printf("\nseed: %d\n", seed);
				auto MDP = generate_random_MDP_with_variable_parameters(S, A_num, action_prob, non_zero_transition, upper_reward, seed);
				R_type R = get<0>(MDP);
				A_type A = get<1>(MDP);
				P_type P = get<2>(MDP);

				//iterations test printing
				float R_max = find_max_R(R);
				float R_min = find_min_R(R);
				float iterations_bound = log(R_max /((1.0 - gamma) * epsilon)) / log(1.0 / gamma);
				printf("R_max is %f\n", R_max);
				printf("R_min is %f\n", R_min);
				printf("upper bound is %f\n", R_max / (1.0 - gamma));
				printf("lower bound is %f\n", R_min / (1.0 - gamma));
				printf("iterations bound: %f\n", iterations_bound);

				//VI testing
				//TODO: make method that takes arguments and performs the writing of data, takes filename, MDP etc. as argument
				A_type A1 = copy_A(A);
				auto start_VI = high_resolution_clock::now();

				V_type V_approx_solution_tuple = value_iteration(S, R, A1, P, gamma, epsilon);
                vector<double> V_approx_solution = get<0>(V_approx_solution_tuple);
                int V_approx_solution_iterations = get<1>(V_approx_solution_tuple);
		
				printf("VI iterations: %d\n", V_approx_solution_iterations);

				auto stop_VI = high_resolution_clock::now();
				auto duration_VI = duration_cast<microseconds>(stop_VI - start_VI);

				stringstream_VI << to_string(S) << " " << V_approx_solution_iterations << endl;

				//VIU testing
				A_type A6 = copy_A(A);
				auto start_VIU = high_resolution_clock::now();

				V_type V_approx_solution_upper_tuple = value_iteration_upper(S, R, A6, P, gamma, epsilon);
				vector<double> V_approx_solution_upper = get<0>(V_approx_solution_upper_tuple);
				int V_approx_solution_upper_iterations = get<1>(V_approx_solution_upper_tuple);
		
				printf("VIU iterations: %d\n", V_approx_solution_upper_iterations);

				auto stop_VIU = high_resolution_clock::now();
				auto duration_VIU = duration_cast<microseconds>(stop_VIU - start_VIU);

				stringstream_VIU << to_string(S) << " " << V_approx_solution_upper_iterations << endl;

				//VIH testing
				A_type A2 = copy_A(A);
				auto start_VIH = high_resolution_clock::now();

				V_type V_heap_approx_tuple = value_iteration_with_heap(S, R, A2, P, gamma, epsilon);
				vector<double> V_heap_approx = get<0>(V_heap_approx_tuple);
				int V_heap_approx_iterations = get<1>(V_heap_approx_tuple);
		
				printf("VIH iterations: %d\n", V_heap_approx_iterations);

				auto stop_VIH = high_resolution_clock::now();
				auto duration_VIH = duration_cast<microseconds>(stop_VIH - start_VIH);

				stringstream_VIH << to_string(S) << " " << V_heap_approx_iterations << endl;

				//BVI
				A_type A3 = copy_A(A);
				auto start_BVI = high_resolution_clock::now();

				V_type V_bounded_approx_solution_tuple = bounded_value_iteration(S, R, A3, P, gamma, epsilon);
				vector<double> V_bounded_approx_solution = get<0>(V_bounded_approx_solution_tuple);
				int V_bounded_approx_solution_iterations = get<1>(V_bounded_approx_solution_tuple);
		
				printf("BVI iterations: %d\n", V_bounded_approx_solution_iterations);

				auto stop_BVI = high_resolution_clock::now();
				auto duration_BVI = duration_cast<microseconds>(stop_BVI - start_BVI);
				
				stringstream_BVI << to_string(S) << " " << V_bounded_approx_solution_iterations << endl;

				//VIAE
				A_type A4 = copy_A(A);
				auto start_VIAE = high_resolution_clock::now();

				V_type V_AE_approx_solution_tuple = value_iteration_action_elimination(S, R, A4, P, gamma, epsilon);
				vector<double> V_AE_approx_solution = get<0>(V_AE_approx_solution_tuple);
				int V_AE_approx_solution_iterations = get<1>(V_AE_approx_solution_tuple);
		
				printf("VIAE iterations: %d\n", V_AE_approx_solution_iterations);

				auto stop_VIAE = high_resolution_clock::now();
				auto duration_VIAE = duration_cast<microseconds>(stop_VIAE - start_VIAE);

				stringstream_VIAE << to_string(S) << " " << V_AE_approx_solution_iterations << endl;

				//VIAEH
				A_type A5 = copy_A(A);
				auto start_VIAEH = high_resolution_clock::now();

				V_type V_AE_H_approx_solution_tuple = value_iteration_action_elimination_heaps(S, R, A5, P, gamma, epsilon);
				vector<double> V_AE_H_approx_solution = get<0>(V_AE_H_approx_solution_tuple);
				int V_AE_H_approx_solution_iterations = get<1>(V_AE_H_approx_solution_tuple);
		
				printf("VIAEH iterations: %d\n", V_AE_H_approx_solution_iterations);

				auto stop_VIAEH = high_resolution_clock::now();
				auto duration_VIAEH = duration_cast<microseconds>(stop_VIAEH - start_VIAEH);

				stringstream_VIAEH << to_string(S) << " " << V_AE_H_approx_solution_iterations << endl;

				//They should in theory all be epsilon from true value, and therefore, at most be 2 * epsilon from each other
				if (abs_max_diff_vectors(V_approx_solution, V_heap_approx) > (2 * epsilon)){
						printf("DIFFERENCE\n");
				}
				if (abs_max_diff_vectors(V_AE_approx_solution, V_AE_H_approx_solution) > (2 * epsilon)){
						printf("DIFFERENCE\n");
				}
				if (abs_max_diff_vectors(V_bounded_approx_solution, V_approx_solution_upper) > (2 * epsilon)){
						printf("DIFFERENCE\n");
				}

		}

		//WRITE ALL DATA TO THEIR RESPECTVIE FILES	
		write_stringstream_to_file(stringstream_VI, output_stream_VI, file_name_VI);
		write_stringstream_to_file(stringstream_VIU, output_stream_VIU, file_name_VIU);
		write_stringstream_to_file(stringstream_BVI, output_stream_BVI, file_name_BVI);
		write_stringstream_to_file(stringstream_VIH, output_stream_VIH, file_name_VIH);
		write_stringstream_to_file(stringstream_VIAE, output_stream_VIAE, file_name_VIAE);
		write_stringstream_to_file(stringstream_VIAEH, output_stream_VIAEH, file_name_VIAEH);
}

//VARYING number of non-zero transition states
void write_meta_data_to_dat_file_number_of_transitions(ostringstream& string_stream, int S, int A_num, double epsilon, double gamma, double upper_reward, double action_prob, int number_of_transitions_starting_value, int number_of_transitions_finishing_value, int number_of_transitions_increment){
		time_t time_now = time(0);
		string_stream << "# META DATA" << endl;
		string_stream << "# " << endl;
		string_stream << "# " << "experiment run at: " << ctime(&time_now);
		string_stream << "# " << endl;
		string_stream << "# " << "gamma = " << gamma << endl;
		string_stream << "# " << "epsilon = " << epsilon << endl;
		string_stream << "# " << "S = " << S << endl;
		string_stream << "# " << "A = " << A_num << endl;
		string_stream << "# " << "action_prob = " << action_prob << endl;
		string_stream << "# " << "upper_reward = " << upper_reward << endl;
		string_stream << "# " << endl;
		string_stream << "# VARIABLE IN THIS EXPERIMENT" << endl;
		string_stream << "# " << "number of transitions is varied from " << number_of_transitions_starting_value << " to " << number_of_transitions_finishing_value << " with " << number_of_transitions_increment << " increment" << endl;
		string_stream << "# " << endl;
		string_stream << "# ACTUAL DATA" << endl;
		string_stream << "# number of transitions | microseconds" << endl;
}

void create_data_tables_number_of_transitions(string filename, int S, int A_num, double epsilon, double gamma, double upper_reward, double action_prob, int max_transitions){

		//the stringstreams to create the test for the files
		ostringstream stringstream_BVI;
		ostringstream stringstream_VI;
		ostringstream stringstream_VIU;
		ostringstream stringstream_VIH;
		ostringstream stringstream_VIAE;
		ostringstream stringstream_VIAEH;

		//the file output objects
		ofstream output_stream_BVI;
		ofstream output_stream_VI;
		ofstream output_stream_VIU;
		ofstream output_stream_VIH;
		ofstream output_stream_VIAE;
		ofstream output_stream_VIAEH;
		
		//set the name of the file to write to
		string file_name_BVI = "data_tables/number_of_transitions/" + filename + "_BVI.dat"; 
		string file_name_VI =  "data_tables/number_of_transitions/" + filename + "_VI.dat";
		string file_name_VIU = "data_tables/number_of_transitions/" + filename + "_VIU.dat";
		string file_name_VIH = "data_tables/number_of_transitions/" + filename + "_VIH.dat";
		string file_name_VIAE = "data_tables/number_of_transitions/" + filename + "_VIAE.dat";
		string file_name_VIAEH = "data_tables/number_of_transitions/" + filename + "_VIAEH.dat";

		int number_of_transitions_starting_value = 10;
		int number_of_transitions_finishing_value = max_transitions;
		int number_of_transitions_increment = 10;

		//write meta data to all stringstreams as first in their respective files
		write_meta_data_to_dat_file_number_of_transitions(stringstream_VI, S, A_num, epsilon, gamma, upper_reward, action_prob, number_of_transitions_starting_value, number_of_transitions_finishing_value, number_of_transitions_increment);
		write_meta_data_to_dat_file_number_of_transitions(stringstream_VIU, S, A_num, epsilon, gamma, upper_reward, action_prob, number_of_transitions_starting_value, number_of_transitions_finishing_value, number_of_transitions_increment);
		write_meta_data_to_dat_file_number_of_transitions(stringstream_VIH, S, A_num, epsilon, gamma, upper_reward, action_prob, number_of_transitions_starting_value, number_of_transitions_finishing_value, number_of_transitions_increment);
		write_meta_data_to_dat_file_number_of_transitions(stringstream_BVI, S, A_num, epsilon, gamma, upper_reward, action_prob, number_of_transitions_starting_value, number_of_transitions_finishing_value, number_of_transitions_increment);
		write_meta_data_to_dat_file_number_of_transitions(stringstream_VIAE, S, A_num, epsilon, gamma, upper_reward, action_prob, number_of_transitions_starting_value, number_of_transitions_finishing_value, number_of_transitions_increment);
		write_meta_data_to_dat_file_number_of_transitions(stringstream_VIAEH, S, A_num, epsilon, gamma, upper_reward, action_prob, number_of_transitions_starting_value, number_of_transitions_finishing_value, number_of_transitions_increment);

		for (int number_of_transitions = number_of_transitions_starting_value; number_of_transitions <= number_of_transitions_finishing_value;  number_of_transitions = number_of_transitions + number_of_transitions_increment){
				
				//status message of experiment
				printf("Beginning iteration number_of_transitions= %d\n", number_of_transitions);

				//GENERATE THE MDP FROM CURRENT TIME SEED
				int seed = time(0);
				auto MDP = generate_random_MDP_with_variable_parameters_fixed_nonzero_trans_states(S, A_num, action_prob, number_of_transitions, upper_reward, seed);
				R_type R = get<0>(MDP);
				A_type A = get<1>(MDP);
				P_type P = get<2>(MDP);

				//VI testing
				//TODO: make method that takes arguments and performs the writing of data, takes filename, MDP etc. as argument
				A_type A1 = copy_A(A);
				auto start_VI = high_resolution_clock::now();

				V_type V_approx_solution_tuple = value_iteration(S, R, A1, P, gamma, epsilon);
                vector<double> V_approx_solution = get<0>(V_approx_solution_tuple);

				auto stop_VI = high_resolution_clock::now();
				auto duration_VI = duration_cast<microseconds>(stop_VI - start_VI);

				stringstream_VI << to_string(number_of_transitions) << " " << duration_VI.count() << endl;

				//VIU testing
				A_type A6 = copy_A(A);
				auto start_VIU = high_resolution_clock::now();

				V_type V_approx_solution_upper_tuple = value_iteration_upper(S, R, A6, P, gamma, epsilon);
				vector<double> V_approx_solution_upper = get<0>(V_approx_solution_upper_tuple);

				auto stop_VIU = high_resolution_clock::now();
				auto duration_VIU = duration_cast<microseconds>(stop_VIU - start_VIU);

				stringstream_VIU << to_string(number_of_transitions) << " " << duration_VIU.count() << endl;

				//VIH testing
				A_type A2 = copy_A(A);
				auto start_VIH = high_resolution_clock::now();

				V_type V_heap_approx_tuple = value_iteration_with_heap(S, R, A2, P, gamma, epsilon);
				vector<double> V_heap_approx = get<0>(V_heap_approx_tuple);

				auto stop_VIH = high_resolution_clock::now();
				auto duration_VIH = duration_cast<microseconds>(stop_VIH - start_VIH);

				stringstream_VIH << to_string(number_of_transitions) << " " << duration_VIH.count() << endl;

				//BVI
				A_type A3 = copy_A(A);
				auto start_BVI = high_resolution_clock::now();

				V_type V_bounded_approx_solution_tuple = bounded_value_iteration(S, R, A3, P, gamma, epsilon);
				vector<double> V_bounded_approx_solution = get<0>(V_bounded_approx_solution_tuple);

				auto stop_BVI = high_resolution_clock::now();
				auto duration_BVI = duration_cast<microseconds>(stop_BVI - start_BVI);
				
				stringstream_BVI << to_string(number_of_transitions) << " " << duration_BVI.count() << endl;

				//VIAE
				A_type A4 = copy_A(A);
				auto start_VIAE = high_resolution_clock::now();

				V_type V_AE_approx_solution_tuple = value_iteration_action_elimination(S, R, A4, P, gamma, epsilon);
				vector<double> V_AE_approx_solution = get<0>(V_AE_approx_solution_tuple);

				auto stop_VIAE = high_resolution_clock::now();
				auto duration_VIAE = duration_cast<microseconds>(stop_VIAE - start_VIAE);

				stringstream_VIAE << to_string(number_of_transitions) << " " << duration_VIAE.count() << endl;

				//VIAEH
				A_type A5 = copy_A(A);
				auto start_VIAEH = high_resolution_clock::now();

				V_type V_AE_H_approx_solution_tuple = value_iteration_action_elimination_heaps(S, R, A5, P, gamma, epsilon);
				vector<double> V_AE_H_approx_solution = get<0>(V_AE_H_approx_solution_tuple);

				auto stop_VIAEH = high_resolution_clock::now();
				auto duration_VIAEH = duration_cast<microseconds>(stop_VIAEH - start_VIAEH);

				stringstream_VIAEH << to_string(number_of_transitions) << " " << duration_VIAEH.count() << endl;

				//They should in theory all be epsilon from true value, and therefore, at most be 2 * epsilon from each other
				if (abs_max_diff_vectors(V_approx_solution, V_heap_approx) > (2 * epsilon)){
						printf("DIFFERENCE\n");
				}
				if (abs_max_diff_vectors(V_AE_approx_solution, V_AE_H_approx_solution) > (2 * epsilon)){
						printf("DIFFERENCE\n");
				}
				if (abs_max_diff_vectors(V_bounded_approx_solution, V_approx_solution_upper) > (2 * epsilon)){
						printf("DIFFERENCE\n");
				}

		}

		//WRITE ALL DATA TO THEIR RESPECTVIE FILES	
		write_stringstream_to_file(stringstream_VI, output_stream_VI, file_name_VI);
		write_stringstream_to_file(stringstream_VIU, output_stream_VIU, file_name_VIU);
		write_stringstream_to_file(stringstream_BVI, output_stream_BVI, file_name_BVI);
		write_stringstream_to_file(stringstream_VIH, output_stream_VIH, file_name_VIH);
		write_stringstream_to_file(stringstream_VIAE, output_stream_VIAE, file_name_VIAE);
		write_stringstream_to_file(stringstream_VIAEH, output_stream_VIAEH, file_name_VIAEH);

}

//VARYING NUMBER OF ACTIONS
void write_meta_data_to_dat_file_number_of_actions_iterations(ostringstream& string_stream, int S, double epsilon, double gamma, double upper_reward, double non_zero_transition, double action_prob, int A_starting_value, int A_finishing_value, int A_increment){
		time_t time_now = time(0);
		string_stream << "# META DATA" << endl;
		string_stream << "# " << endl;
		string_stream << "# " << "experiment run at: " << ctime(&time_now);
		string_stream << "# " << endl;
		string_stream << "# " << "gamma = " << gamma << endl;
		string_stream << "# " << "epsilon = " << epsilon << endl;
		string_stream << "# " << "S = " << S << endl;
		string_stream << "# " << "non_zero_transition = " << non_zero_transition << endl;
		string_stream << "# " << "upper_reward = " << upper_reward << endl;
		string_stream << "# " << "action_prob = " << action_prob << endl;
		string_stream << "# " << endl;
		string_stream << "# VARIABLE IN THIS EXPERIMENT" << endl;
		string_stream << "# " << "Number of actions, A, is varied from " << A_starting_value << " to " << A_finishing_value << " with " << A_increment << " increment" << endl;
		string_stream << "# " << endl;
		string_stream << "# ACTUAL DATA" << endl;
		string_stream << "# number of actions | iterations" << endl;
}

void create_data_tables_number_of_actions_iterations(string filename, int S, int A_max, double epsilon, double gamma, double upper_reward, double non_zero_transition){

		//the stringstreams to create the test for the files
		ostringstream stringstream_BVI;
		ostringstream stringstream_VI;
		ostringstream stringstream_VIU;
		ostringstream stringstream_VIH;
		ostringstream stringstream_VIAE;
		ostringstream stringstream_VIAEH;

		//the file output objects
		ofstream output_stream_BVI;
		ofstream output_stream_VI;
		ofstream output_stream_VIU;
		ofstream output_stream_VIH;
		ofstream output_stream_VIAE;
		ofstream output_stream_VIAEH;
		
		//set the name of the file to write to
		string file_name_BVI = "data_tables/number_of_actions_iterations/" + filename + "_BVI.dat"; 
		string file_name_VI =  "data_tables/number_of_actions_iterations/" + filename + "_VI.dat";
		string file_name_VIU =  "data_tables/number_of_actions_iterations/" + filename + "_VIU.dat";
		string file_name_VIH = "data_tables/number_of_actions_iterations/" + filename + "_VIH.dat";
		string file_name_VIAE = "data_tables/number_of_actions_iterations/" + filename + "_VIAE.dat";
		string file_name_VIAEH = "data_tables/number_of_actions_iterations/" + filename + "_VIAEH.dat";

		//The varying parameters
		int A_starting_value = 50;
		int A_finishing_value = A_max;
		int A_increment = 50;

		//hardcoded parameter
		double action_prob = 1.0;

		//write meta data to all stringstreams as first in their respective files
		
		write_meta_data_to_dat_file_number_of_actions_iterations(stringstream_VI, S, epsilon, gamma, upper_reward, non_zero_transition, action_prob, A_starting_value, A_finishing_value, A_increment);
		write_meta_data_to_dat_file_number_of_actions_iterations(stringstream_VIU, S, epsilon, gamma, upper_reward, non_zero_transition, action_prob, A_starting_value, A_finishing_value, A_increment);
		write_meta_data_to_dat_file_number_of_actions_iterations(stringstream_BVI, S, epsilon, gamma, upper_reward, non_zero_transition, action_prob, A_starting_value, A_finishing_value, A_increment);
		write_meta_data_to_dat_file_number_of_actions_iterations(stringstream_VIH, S, epsilon, gamma, upper_reward, non_zero_transition, action_prob, A_starting_value, A_finishing_value, A_increment);
		write_meta_data_to_dat_file_number_of_actions_iterations(stringstream_VIAE, S, epsilon, gamma, upper_reward, non_zero_transition, action_prob, A_starting_value, A_finishing_value, A_increment);
		write_meta_data_to_dat_file_number_of_actions_iterations(stringstream_VIAEH, S, epsilon, gamma, upper_reward, non_zero_transition, action_prob, A_starting_value, A_finishing_value, A_increment);

		for (int A_num = A_starting_value; A_num <= A_finishing_value; A_num = A_num + A_increment){
				
				printf("Beginning iteration A_num = %d\n", A_num);

				//GENERATE THE MDP
				int seed = time(0);
				printf("\nseed: %d\n", seed);
				auto MDP = generate_random_MDP_with_variable_parameters(S, A_num, action_prob, non_zero_transition, upper_reward, seed);
				R_type R = get<0>(MDP);
				A_type A = get<1>(MDP);
				P_type P = get<2>(MDP);

				//iterations test printing
				float R_max = find_max_R(R);
				float R_min = find_min_R(R);
				float iterations_bound = log(R_max /((1.0 - gamma) * epsilon)) / log(1.0 / gamma);
				printf("R_max is %f\n", R_max);
				printf("R_min is %f\n", R_min);
				printf("upper bound is %f\n", R_max / (1.0 - gamma));
				printf("lower bound is %f\n", R_min / (1.0 - gamma));
				printf("iterations bound: %f\n", iterations_bound);

				//VI testing
				//TODO: make method that takes arguments and performs the writing of data, takes filename, MDP etc. as argument
				A_type A1 = copy_A(A);
				auto start_VI = high_resolution_clock::now();

				V_type V_approx_solution_tuple = value_iteration(S, R, A1, P, gamma, epsilon);
                vector<double> V_approx_solution = get<0>(V_approx_solution_tuple);
                int V_approx_solution_iterations = get<1>(V_approx_solution_tuple);
		
				printf("VI iterations: %d\n", V_approx_solution_iterations);

				auto stop_VI = high_resolution_clock::now();
				auto duration_VI = duration_cast<microseconds>(stop_VI - start_VI);

				stringstream_VI << to_string(A_num) << " " << V_approx_solution_iterations << endl;

				//VIU testing
				A_type A6 = copy_A(A);
				auto start_VIU = high_resolution_clock::now();

				V_type V_approx_solution_upper_tuple = value_iteration_upper(S, R, A6, P, gamma, epsilon);
				vector<double> V_approx_solution_upper = get<0>(V_approx_solution_upper_tuple);
				int V_approx_solution_upper_iterations = get<1>(V_approx_solution_upper_tuple);
		
				printf("VIU iterations: %d\n", V_approx_solution_upper_iterations);

				auto stop_VIU = high_resolution_clock::now();
				auto duration_VIU = duration_cast<microseconds>(stop_VIU - start_VIU);

				stringstream_VIU << to_string(A_num) << " " << V_approx_solution_upper_iterations << endl;

				//VIH testing
				A_type A2 = copy_A(A);
				auto start_VIH = high_resolution_clock::now();

				V_type V_heap_approx_tuple = value_iteration_with_heap(S, R, A2, P, gamma, epsilon);
				vector<double> V_heap_approx = get<0>(V_heap_approx_tuple);
				int V_heap_approx_iterations = get<1>(V_heap_approx_tuple);
		
				printf("VIH iterations: %d\n", V_heap_approx_iterations);

				auto stop_VIH = high_resolution_clock::now();
				auto duration_VIH = duration_cast<microseconds>(stop_VIH - start_VIH);

				stringstream_VIH << to_string(A_num) << " " << V_heap_approx_iterations << endl;

				//BVI
				A_type A3 = copy_A(A);
				auto start_BVI = high_resolution_clock::now();

				V_type V_bounded_approx_solution_tuple = bounded_value_iteration(S, R, A3, P, gamma, epsilon);
				vector<double> V_bounded_approx_solution = get<0>(V_bounded_approx_solution_tuple);
				int V_bounded_approx_solution_iterations = get<1>(V_bounded_approx_solution_tuple);
		
				printf("BVI iterations: %d\n", V_bounded_approx_solution_iterations);

				auto stop_BVI = high_resolution_clock::now();
				auto duration_BVI = duration_cast<microseconds>(stop_BVI - start_BVI);
				
				stringstream_BVI << to_string(A_num) << " " << V_bounded_approx_solution_iterations << endl;

				//VIAE
				A_type A4 = copy_A(A);
				auto start_VIAE = high_resolution_clock::now();

				V_type V_AE_approx_solution_tuple = value_iteration_action_elimination(S, R, A4, P, gamma, epsilon);
				vector<double> V_AE_approx_solution = get<0>(V_AE_approx_solution_tuple);
				int V_AE_approx_solution_iterations = get<1>(V_AE_approx_solution_tuple);
		
				printf("VIAE iterations: %d\n", V_AE_approx_solution_iterations);

				auto stop_VIAE = high_resolution_clock::now();
				auto duration_VIAE = duration_cast<microseconds>(stop_VIAE - start_VIAE);

				stringstream_VIAE << to_string(A_num) << " " << V_AE_approx_solution_iterations << endl;

				//VIAEH
				A_type A5 = copy_A(A);
				auto start_VIAEH = high_resolution_clock::now();

				V_type V_AE_H_approx_solution_tuple = value_iteration_action_elimination_heaps(S, R, A5, P, gamma, epsilon);
				vector<double> V_AE_H_approx_solution = get<0>(V_AE_H_approx_solution_tuple);
				int V_AE_H_approx_solution_iterations = get<1>(V_AE_H_approx_solution_tuple);
		
				printf("VIAEH iterations: %d\n", V_AE_H_approx_solution_iterations);

				auto stop_VIAEH = high_resolution_clock::now();
				auto duration_VIAEH = duration_cast<microseconds>(stop_VIAEH - start_VIAEH);

				stringstream_VIAEH << to_string(A_num) << " " << V_AE_H_approx_solution_iterations << endl;

				//They should in theory all be epsilon from true value, and therefore, at most be 2 * epsilon from each other
				if (abs_max_diff_vectors(V_approx_solution, V_heap_approx) > (2 * epsilon)){
						printf("DIFFERENCE\n");
				}
				if (abs_max_diff_vectors(V_AE_approx_solution, V_AE_H_approx_solution) > (2 * epsilon)){
						printf("DIFFERENCE\n");
				}
				if (abs_max_diff_vectors(V_bounded_approx_solution, V_approx_solution_upper) > (2 * epsilon)){
						printf("DIFFERENCE\n");
				}

		}

		//WRITE ALL DATA TO THEIR RESPECTVIE FILES	
		write_stringstream_to_file(stringstream_VI, output_stream_VI, file_name_VI);
		write_stringstream_to_file(stringstream_VIU, output_stream_VIU, file_name_VIU);
		write_stringstream_to_file(stringstream_BVI, output_stream_BVI, file_name_BVI);
		write_stringstream_to_file(stringstream_VIH, output_stream_VIH, file_name_VIH);
		write_stringstream_to_file(stringstream_VIAE, output_stream_VIAE, file_name_VIAE);
		write_stringstream_to_file(stringstream_VIAEH, output_stream_VIAEH, file_name_VIAEH);
}

//VARYING THE REWARD SPACE
void write_meta_data_to_dat_file_max_reward(ostringstream& string_stream, int S, int A_num, double epsilon, double gamma, double non_zero_transition, double action_prob, double max_reward_starting_value, double max_reward_finishing_value, double max_reward_increment){
		time_t time_now = time(0);
		string_stream << "# META DATA" << endl;
		string_stream << "# " << endl;
		string_stream << "# " << "experiment run at: " << ctime(&time_now);
		string_stream << "# " << endl;
		string_stream << "# " << "gamma = " << gamma << endl;
		string_stream << "# " << "epsilon = " << epsilon << endl;
		string_stream << "# " << "S = " << S << endl;
		string_stream << "# " << "A = " << A_num << endl;
		string_stream << "# " << "non_zero_transition = " << non_zero_transition << endl;
		string_stream << "# " << "action_prob = " << action_prob << endl;
		string_stream << "# " << endl;
		string_stream << "# VARIABLE IN THIS EXPERIMENT" << endl;
		string_stream << "# " << "max_reward is varied from " << max_reward_starting_value << " to " << max_reward_finishing_value << " with " << max_reward_increment << " increment" << endl;
		string_stream << "# " << endl;
		string_stream << "# ACTUAL DATA" << endl;
		string_stream << "# action_prob | microseconds" << endl;
}

void create_data_tables_max_reward(string filename, int S, int A_num, double epsilon, double gamma, double action_prob, double max_reward_finishing_value, double non_zero_transition){

		//the stringstreams to create the test for the files
		ostringstream stringstream_BVI;
		ostringstream stringstream_VI;
		ostringstream stringstream_VIU;
		ostringstream stringstream_VIH;
		ostringstream stringstream_VIAE;
		ostringstream stringstream_VIAEH;

		//the file output objects
		ofstream output_stream_BVI;
		ofstream output_stream_VI;
		ofstream output_stream_VIU;
		ofstream output_stream_VIH;
		ofstream output_stream_VIAE;
		ofstream output_stream_VIAEH;
		
		//set the name of the file to write to
		string file_name_BVI = "data_tables/max_reward/" + filename + "_BVI.dat"; 
		string file_name_VI =  "data_tables/max_reward/" + filename + "_VI.dat";
		string file_name_VIU = "data_tables/max_reward/" + filename + "_VIU.dat";
		string file_name_VIH = "data_tables/max_reward/" + filename + "_VIH.dat";
		string file_name_VIAE = "data_tables/max_reward/" + filename + "_VIAE.dat";
		string file_name_VIAEH = "data_tables/max_reward/" + filename + "_VIAEH.dat";

		double max_reward_starting_value = 100.0;
		//double max_reward_finishing_value = max_reward_finishing_value;
		double max_reward_increment = 100.0;

		//write meta data to all stringstreams as first in their respective files
		write_meta_data_to_dat_file_max_reward(stringstream_VI, S, A_num, epsilon, gamma, non_zero_transition, action_prob, max_reward_starting_value, max_reward_finishing_value, max_reward_increment);
		write_meta_data_to_dat_file_max_reward(stringstream_VIU, S, A_num, epsilon, gamma, non_zero_transition, action_prob, max_reward_starting_value, max_reward_finishing_value, max_reward_increment);
		write_meta_data_to_dat_file_max_reward(stringstream_VIH, S, A_num, epsilon, gamma, non_zero_transition, action_prob, max_reward_starting_value, max_reward_finishing_value, max_reward_increment);
		write_meta_data_to_dat_file_max_reward(stringstream_BVI, S, A_num, epsilon, gamma, non_zero_transition, action_prob, max_reward_starting_value, max_reward_finishing_value, max_reward_increment);
		write_meta_data_to_dat_file_max_reward(stringstream_VIAE, S, A_num, epsilon, gamma, non_zero_transition, action_prob, max_reward_starting_value, max_reward_finishing_value, max_reward_increment);
		write_meta_data_to_dat_file_max_reward(stringstream_VIAEH, S, A_num, epsilon, gamma, non_zero_transition, action_prob, max_reward_starting_value, max_reward_finishing_value, max_reward_increment);

		for (double max_reward = max_reward_starting_value; max_reward <= max_reward_finishing_value; max_reward = max_reward + max_reward_increment){
				
				//status message of experiment
				printf("Beginning iteration max_reward = %f\n", max_reward);

				//GENERATE THE MDP FROM CURRENT TIME SEED
				int seed = time(0);
				auto MDP = generate_random_MDP_with_variable_parameters(S, A_num, action_prob, non_zero_transition, max_reward, seed);
				R_type R = get<0>(MDP);
				A_type A = get<1>(MDP);
				P_type P = get<2>(MDP);

				//iterations test printing
				float R_max = find_max_R(R);
				float R_min = find_min_R(R);
				float iterations_bound = log(R_max /((1.0 - gamma) * epsilon)) / log(1.0 / gamma);
				printf("R_max is %f\n", R_max);
				printf("R_min is %f\n", R_min);
				printf("upper bound is %f\n", R_max / (1.0 - gamma));
				printf("lower bound is %f\n", R_min / (1.0 - gamma));
				printf("iterations bound: %f\n", iterations_bound);
				

				//VI testing
				A_type A1 = copy_A(A);
				auto start_VI = high_resolution_clock::now();

				V_type V_approx_solution_tuple = value_iteration(S, R, A1, P, gamma, epsilon);
                vector<double> V_approx_solution = get<0>(V_approx_solution_tuple);

				auto stop_VI = high_resolution_clock::now();
				auto duration_VI = duration_cast<microseconds>(stop_VI - start_VI);

				stringstream_VI << to_string(max_reward) << " " << duration_VI.count() << endl;

				//VIU testing
				A_type A6 = copy_A(A);
				auto start_VIU = high_resolution_clock::now();

				V_type V_approx_solution_upper_tuple = value_iteration_upper(S, R, A6, P, gamma, epsilon);
				vector<double> V_approx_solution_upper = get<0>(V_approx_solution_upper_tuple);

				auto stop_VIU = high_resolution_clock::now();
				auto duration_VIU = duration_cast<microseconds>(stop_VIU - start_VIU);

				stringstream_VIU << to_string(max_reward) << " " << duration_VIU.count() << endl;

				//VIH testing
				A_type A2 = copy_A(A);
				auto start_VIH = high_resolution_clock::now();

				V_type V_heap_approx_tuple = value_iteration_with_heap(S, R, A2, P, gamma, epsilon);
				vector<double> V_heap_approx = get<0>(V_heap_approx_tuple);

				auto stop_VIH = high_resolution_clock::now();
				auto duration_VIH = duration_cast<microseconds>(stop_VIH - start_VIH);

				stringstream_VIH << to_string(max_reward) << " " << duration_VIH.count() << endl;

				//BVI
				A_type A3 = copy_A(A);
				auto start_BVI = high_resolution_clock::now();

				V_type V_bounded_approx_solution_tuple = bounded_value_iteration(S, R, A3, P, gamma, epsilon);
				vector<double> V_bounded_approx_solution = get<0>(V_bounded_approx_solution_tuple);

				auto stop_BVI = high_resolution_clock::now();
				auto duration_BVI = duration_cast<microseconds>(stop_BVI - start_BVI);
				
				stringstream_BVI << to_string(max_reward) << " " << duration_BVI.count() << endl;

				//VIAE
				A_type A4 = copy_A(A);
				auto start_VIAE = high_resolution_clock::now();

				V_type V_AE_approx_solution_tuple = value_iteration_action_elimination(S, R, A4, P, gamma, epsilon);
				vector<double> V_AE_approx_solution = get<0>(V_AE_approx_solution_tuple);

				auto stop_VIAE = high_resolution_clock::now();
				auto duration_VIAE = duration_cast<microseconds>(stop_VIAE - start_VIAE);

				stringstream_VIAE << to_string(max_reward) << " " << duration_VIAE.count() << endl;

				//VIAEH
				A_type A5 = copy_A(A);
				auto start_VIAEH = high_resolution_clock::now();

				V_type V_AE_H_approx_solution_tuple = value_iteration_action_elimination_heaps(S, R, A5, P, gamma, epsilon);
				vector<double> V_AE_H_approx_solution = get<0>(V_AE_H_approx_solution_tuple);

				auto stop_VIAEH = high_resolution_clock::now();
				auto duration_VIAEH = duration_cast<microseconds>(stop_VIAEH - start_VIAEH);

				stringstream_VIAEH << to_string(max_reward) << " " << duration_VIAEH.count() << endl;

				//They should in theory all be epsilon from true value, and therefore, at most be 2 * epsilon from each other
				if (abs_max_diff_vectors(V_approx_solution, V_heap_approx) > (2 * epsilon)){
						printf("DIFFERENCE\n");
				}
				if (abs_max_diff_vectors(V_AE_approx_solution, V_AE_H_approx_solution) > (2 * epsilon)){
						printf("DIFFERENCE\n");
				}
				if (abs_max_diff_vectors(V_bounded_approx_solution, V_approx_solution_upper) > (2 * epsilon)){
						printf("DIFFERENCE\n");
				}

		}

		//WRITE ALL DATA TO THEIR RESPECTVIE FILES	
		write_stringstream_to_file(stringstream_VI, output_stream_VI, file_name_VI);
		write_stringstream_to_file(stringstream_VIU, output_stream_VIU, file_name_VIU);
		write_stringstream_to_file(stringstream_BVI, output_stream_BVI, file_name_BVI);
		write_stringstream_to_file(stringstream_VIH, output_stream_VIH, file_name_VIH);
		write_stringstream_to_file(stringstream_VIAE, output_stream_VIAE, file_name_VIAE);
		write_stringstream_to_file(stringstream_VIAEH, output_stream_VIAEH, file_name_VIAEH);

}

//VARYING NUMBER OF ACTIONS - Iterations until convergence plot
void write_meta_data_to_dat_file_number_of_actions_first_convergence_iteration(ostringstream& string_stream, int S, double epsilon, double gamma, double upper_reward, double non_zero_transition, double action_prob, int A_starting_value, int A_finishing_value, int A_increment){
		time_t time_now = time(0);
		string_stream << "# META DATA" << endl;
		string_stream << "# " << endl;
		string_stream << "# " << "experiment run at: " << ctime(&time_now);
		string_stream << "# " << endl;
		string_stream << "# " << "gamma = " << gamma << endl;
		string_stream << "# " << "epsilon = " << epsilon << endl;
		string_stream << "# " << "S = " << S << endl;
		string_stream << "# " << "non_zero_transition = " << non_zero_transition << endl;
		string_stream << "# " << "upper_reward = " << upper_reward << endl;
		string_stream << "# " << "action_prob = " << action_prob << endl;
		string_stream << "# " << endl;
		string_stream << "# VARIABLE IN THIS EXPERIMENT" << endl;
		string_stream << "# " << "Number of actions, A, is varied from " << A_starting_value << " to " << A_finishing_value << " with " << A_increment << " increment" << endl;
		string_stream << "# " << endl;
		string_stream << "# ACTUAL DATA" << endl;
		string_stream << "# number of actions | iterations" << endl;
}

void create_data_tables_number_of_actions_first_convergence_iteration(string filename, int S, int A_max, double epsilon, double gamma, double upper_reward, double non_zero_transition){

		//the stringstreams to create the test for the files
		ostringstream stringstream_BVI;
		ostringstream stringstream_BVI_max;
		ostringstream stringstream_VI;
		ostringstream stringstream_VI_max;
		ostringstream stringstream_VIU;
		ostringstream stringstream_theoretical;

		//the file output objects
		ofstream output_stream_BVI;
		ofstream output_stream_BVI_max;
		ofstream output_stream_VI;
		ofstream output_stream_VI_max;
		ofstream output_stream_VIU;
		ofstream output_stream_theoretical;
		
		//set the name of the file to write to
		string file_name_BVI = "data_tables/number_of_actions_first_convergence_iteration/" + filename + "_BVI.dat"; 
		string file_name_BVI_max = "data_tables/number_of_actions_first_convergence_iteration/" + filename + "_BVI_max.dat"; 
		string file_name_VI =  "data_tables/number_of_actions_first_convergence_iteration/" + filename + "_VI.dat";
		string file_name_VI_max =  "data_tables/number_of_actions_first_convergence_iteration/" + filename + "_VI_max.dat";
		string file_name_VIU =  "data_tables/number_of_actions_first_convergence_iteration/" + filename + "_VIU.dat";
		string file_name_theoretical =  "data_tables/number_of_actions_first_convergence_iteration/" + filename + "_theoretical.dat";

		//The varying parameters
		int A_starting_value = 50;
		int A_finishing_value = A_max;
		int A_increment = 50;

		//hardcoded parameter
		double action_prob = 1.0;

		//write meta data to all stringstreams as first in their respective files
		write_meta_data_to_dat_file_number_of_actions_first_convergence_iteration(stringstream_VI, S, epsilon, gamma, upper_reward, non_zero_transition, action_prob, A_starting_value, A_finishing_value, A_increment);
		write_meta_data_to_dat_file_number_of_actions_first_convergence_iteration(stringstream_VI_max, S, epsilon, gamma, upper_reward, non_zero_transition, action_prob, A_starting_value, A_finishing_value, A_increment);
		write_meta_data_to_dat_file_number_of_actions_first_convergence_iteration(stringstream_VIU, S, epsilon, gamma, upper_reward, non_zero_transition, action_prob, A_starting_value, A_finishing_value, A_increment);
		write_meta_data_to_dat_file_number_of_actions_first_convergence_iteration(stringstream_BVI, S, epsilon, gamma, upper_reward, non_zero_transition, action_prob, A_starting_value, A_finishing_value, A_increment);
		write_meta_data_to_dat_file_number_of_actions_first_convergence_iteration(stringstream_BVI_max, S, epsilon, gamma, upper_reward, non_zero_transition, action_prob, A_starting_value, A_finishing_value, A_increment);
		write_meta_data_to_dat_file_number_of_actions_first_convergence_iteration(stringstream_theoretical, S, epsilon, gamma, upper_reward, non_zero_transition, action_prob, A_starting_value, A_finishing_value, A_increment);

		for (int A_num = A_starting_value; A_num <= A_finishing_value; A_num = A_num + A_increment){
				
				printf("Beginning iteration A_num = %d\n", A_num);

				//GENERATE THE MDP
				int seed = time(0);
				auto MDP = generate_random_MDP_with_variable_parameters(S, A_num, 1.0, non_zero_transition, upper_reward, seed);
				R_type R = get<0>(MDP);
				A_type A = get<1>(MDP);
				P_type P = get<2>(MDP);

				float R_max = find_max_R(R);

				//theoretical bound from single instance
				float theoretical_single_instance_bound = log(R_max /((1.0 - gamma) * epsilon)) / log(1.0 / gamma);
				
				//the convergence criterias
				vector<int> V_convergence_bounds = first_convergence_iteration(S, R, A, P, gamma, epsilon);

				stringstream_BVI << to_string(A_num) << " " << V_convergence_bounds[0] << endl;
				stringstream_VIU << to_string(A_num) << " " << V_convergence_bounds[1] << endl;
				stringstream_VI << to_string(A_num) << " " << V_convergence_bounds[2] << endl;
				stringstream_BVI_max << to_string(A_num) << " " << V_convergence_bounds[3] << endl;
				stringstream_theoretical << to_string(A_num) << " " << theoretical_single_instance_bound << endl;
		}

		//WRITE ALL DATA TO THEIR RESPECTVIE FILES	
		write_stringstream_to_file(stringstream_VI, output_stream_VI, file_name_VI);
		write_stringstream_to_file(stringstream_VIU, output_stream_VIU, file_name_VIU);
		write_stringstream_to_file(stringstream_BVI, output_stream_BVI, file_name_BVI);
		write_stringstream_to_file(stringstream_BVI_max, output_stream_BVI_max, file_name_BVI_max);
		write_stringstream_to_file(stringstream_theoretical, output_stream_BVI, file_name_theoretical);
}

//WORK PER ITERATION
void write_meta_data_to_dat_file_work_per_iteration(ostringstream& string_stream, int S, int A_num, double epsilon, double gamma, int number_of_non_zero_transition, double action_prob){
		time_t time_now = time(0);
		string_stream << "# META DATA" << endl;
		string_stream << "# " << endl;
		string_stream << "# " << "experiment run at: " << ctime(&time_now);
		string_stream << "# " << endl;
		string_stream << "# " << "gamma = " << gamma << endl;
		string_stream << "# " << "epsilon = " << epsilon << endl;
		string_stream << "# " << "S = " << S << endl;
		string_stream << "# " << "A = " << A_num << endl;
		string_stream << "# " << "number_of_non_zero_transition = " << number_of_non_zero_transition << endl;
		string_stream << "# " << "action_prob = " << action_prob << endl;
		string_stream << "# " << endl;
		string_stream << "# ACTUAL DATA" << endl;
		string_stream << "# iteration number | microseconds" << endl;
}

void create_data_tables_work_per_iteration(string filename, int S, int A_num, double epsilon, double gamma, double action_prob, int non_zero_transition, double mean, double variance){

		//FOR WORK PER ITERATION
		//the stringstreams to create the test for the files
		ostringstream stringstream_BVI;
		ostringstream stringstream_VI;
		ostringstream stringstream_VIU;
		ostringstream stringstream_VIH;
		ostringstream stringstream_VIAE;
		ostringstream stringstream_VIAEH;
		ostringstream stringstream_VIAEHL;
		ostringstream stringstream_BAO;

		//the file output objects
		ofstream output_stream_BVI;
		ofstream output_stream_VI;
		ofstream output_stream_VIU;
		ofstream output_stream_VIH;
		ofstream output_stream_VIAE;
		ofstream output_stream_VIAEH;
		ofstream output_stream_VIAEHL;
		ofstream output_stream_BAO;
		
		//set the name of the file to write to
		string file_name_BVI = "data_tables/work_per_iteration/" + filename + "_BVI.dat"; 
		string file_name_VI =  "data_tables/work_per_iteration/" + filename + "_VI.dat";
		string file_name_VIU = "data_tables/work_per_iteration/" + filename + "_VIU.dat";
		string file_name_VIH = "data_tables/work_per_iteration/" + filename + "_VIH.dat";
		string file_name_VIAE = "data_tables/work_per_iteration/" + filename + "_VIAE.dat";
		string file_name_VIAEH = "data_tables/work_per_iteration/" + filename + "_VIAEH.dat";
		string file_name_VIAEHL = "data_tables/work_per_iteration/" + filename + "_VIAEHL.dat";
		string file_name_BAO = "data_tables/work_per_iteration/" + filename + "_BAO.dat";

		//write meta data to all stringstreams as first in their respective files
		write_meta_data_to_dat_file_work_per_iteration(stringstream_BVI, S, A_num, epsilon, gamma, non_zero_transition, action_prob);
		write_meta_data_to_dat_file_work_per_iteration(stringstream_VI, S, A_num, epsilon, gamma, non_zero_transition, action_prob);
		write_meta_data_to_dat_file_work_per_iteration(stringstream_VIU, S, A_num, epsilon, gamma, non_zero_transition, action_prob);
		write_meta_data_to_dat_file_work_per_iteration(stringstream_VIH, S, A_num, epsilon, gamma, non_zero_transition, action_prob);
		write_meta_data_to_dat_file_work_per_iteration(stringstream_VIAE, S, A_num, epsilon, gamma, non_zero_transition, action_prob);
		write_meta_data_to_dat_file_work_per_iteration(stringstream_VIAEH, S, A_num, epsilon, gamma, non_zero_transition, action_prob);
		write_meta_data_to_dat_file_work_per_iteration(stringstream_VIAEHL, S, A_num, epsilon, gamma, non_zero_transition, action_prob);
		write_meta_data_to_dat_file_work_per_iteration(stringstream_BAO, S, A_num, epsilon, gamma, non_zero_transition, action_prob);
		
		//FOR ACCUMULATED WORK
		//the stringstreams to create the test for the files
		ostringstream stringstream_accum_BVI;
		ostringstream stringstream_accum_VI;
		ostringstream stringstream_accum_VIU;
		ostringstream stringstream_accum_VIH;
		ostringstream stringstream_accum_VIAE;
		ostringstream stringstream_accum_VIAEH;
		ostringstream stringstream_accum_VIAEHL;
		ostringstream stringstream_accum_BAO;

		//the file output objects
		ofstream output_stream_accum_BVI;
		ofstream output_stream_accum_VI;
		ofstream output_stream_accum_VIU;
		ofstream output_stream_accum_VIH;
		ofstream output_stream_accum_VIAE;
		ofstream output_stream_accum_VIAEH;
		ofstream output_stream_accum_VIAEHL;
		ofstream output_stream_accum_BAO;
		
		//set the name of the file to write to
		string file_name_accum_BVI = "data_tables/work_per_iteration_accum/" + filename + "_accum_BVI.dat"; 
		string file_name_accum_VI =  "data_tables/work_per_iteration_accum/" + filename + "_accum_VI.dat";
		string file_name_accum_VIU = "data_tables/work_per_iteration_accum/" + filename + "_accum_VIU.dat";
		string file_name_accum_VIH = "data_tables/work_per_iteration_accum/" + filename + "_accum_VIH.dat";
		string file_name_accum_VIAE = "data_tables/work_per_iteration_accum/" + filename + "_accum_VIAE.dat";
		string file_name_accum_VIAEH = "data_tables/work_per_iteration_accum/" + filename + "_accum_VIAEH.dat";
		string file_name_accum_VIAEHL = "data_tables/work_per_iteration_accum/" + filename + "_accum_VIAEHL.dat";
		string file_name_accum_BAO = "data_tables/work_per_iteration_accum/" + filename + "_accum_BAO.dat";

		//write meta data to all stringstreams as first in their respective files
		write_meta_data_to_dat_file_work_per_iteration(stringstream_accum_VI, S, A_num, epsilon, gamma, non_zero_transition, action_prob);
		write_meta_data_to_dat_file_work_per_iteration(stringstream_accum_VIU, S, A_num, epsilon, gamma, non_zero_transition, action_prob);
		write_meta_data_to_dat_file_work_per_iteration(stringstream_accum_VIH, S, A_num, epsilon, gamma, non_zero_transition, action_prob);
		write_meta_data_to_dat_file_work_per_iteration(stringstream_accum_VIAE, S, A_num, epsilon, gamma, non_zero_transition, action_prob);
		write_meta_data_to_dat_file_work_per_iteration(stringstream_accum_VIAEH, S, A_num, epsilon, gamma, non_zero_transition, action_prob);
		write_meta_data_to_dat_file_work_per_iteration(stringstream_accum_VIAEHL, S, A_num, epsilon, gamma, non_zero_transition, action_prob);
		write_meta_data_to_dat_file_work_per_iteration(stringstream_accum_BAO, S, A_num, epsilon, gamma, non_zero_transition, action_prob);
		write_meta_data_to_dat_file_work_per_iteration(stringstream_accum_BVI, S, A_num, epsilon, gamma, non_zero_transition, action_prob);

		//BEGIN EXPERIMENTATION
		//GENERATE THE MDP FROM CURRENT TIME SEED
		int seed = time(0);
		printf("seed: %d\n", seed);

		//TODO permament change to normal distribution here?
		//auto MDP = generate_random_MDP_normal_distributed_rewards(S, A_num, action_prob, non_zero_transition, seed, mean, variance);
		auto MDP = generate_random_MDP_exponential_distributed_rewards(S, A_num, action_prob, non_zero_transition, 0.02, seed);
		//auto MDP = generate_random_MDP_with_variable_parameters(S, A_num, action_prob, non_zero_transition, upper_reward, seed);
		R_type R = get<0>(MDP);
		A_type A = get<1>(MDP);
		P_type P = get<2>(MDP);

		//VIAEH
		printf("VIAEH\n");
		A_type A5 = copy_A(A);

		V_type V_AE_H_approx_solution_tuple = value_iteration_action_elimination_heaps(S, R, A5, P, gamma, epsilon);
		vector<double> V_AE_H_approx_solution = get<0>(V_AE_H_approx_solution_tuple);
		int V_AE_H_approx_solution_iterations = get<1>(V_AE_H_approx_solution_tuple);
		vector<microseconds> V_AE_H_approx_solution_work_per_iteration = get<2>(V_AE_H_approx_solution_tuple);

		auto tick_accumulator_VIAEH = V_AE_H_approx_solution_work_per_iteration[0].count();

		for(int iteration = 1; iteration <= V_AE_H_approx_solution_iterations; iteration++) {
				auto iteration_work = V_AE_H_approx_solution_work_per_iteration[iteration].count();
				tick_accumulator_VIAEH = tick_accumulator_VIAEH + iteration_work;

				stringstream_VIAEH << to_string(iteration) << " " << iteration_work << endl;
				stringstream_accum_VIAEH << to_string(iteration) << " " << tick_accumulator_VIAEH << endl;
		}

		//VI testing
		printf("VI\n");
		A_type A1 = copy_A(A);

		V_type V_approx_solution_tuple = value_iteration(S, R, A1, P, gamma, epsilon);
		vector<double> V_approx_solution = get<0>(V_approx_solution_tuple);
		int V_approx_solution_iterations = get<1>(V_approx_solution_tuple);
		vector<microseconds> V_approx_solution_work_per_iteration = get<2>(V_approx_solution_tuple);

		//first entry is zero of the tick type (check this)
		auto tick_accumulator_VI = V_approx_solution_work_per_iteration[0].count();

		for(int iteration = 1; iteration <= V_approx_solution_iterations; iteration++) {
				auto iteration_work = V_approx_solution_work_per_iteration[iteration].count();
				tick_accumulator_VI = tick_accumulator_VI + iteration_work;

				stringstream_VI << to_string(iteration) << " " << iteration_work << endl;
				stringstream_accum_VI << to_string(iteration) << " " << tick_accumulator_VI << endl;
		}

		//VIU testing
		printf("VIU\n");
		A_type A6 = copy_A(A);

		V_type V_approx_solution_upper_tuple = value_iteration_upper(S, R, A6, P, gamma, epsilon);
		vector<double> V_approx_solution_upper = get<0>(V_approx_solution_upper_tuple);
		int V_approx_solution_upper_iterations = get<1>(V_approx_solution_upper_tuple);
		vector<microseconds> V_approx_solution_upper_work_per_iteration = get<2>(V_approx_solution_upper_tuple);

		auto tick_accumulator_VIU = V_approx_solution_upper_work_per_iteration[0].count();
		
		for(int iteration = 1; iteration <= V_approx_solution_upper_iterations; iteration++) {
				auto iteration_work = V_approx_solution_upper_work_per_iteration[iteration].count();
				tick_accumulator_VIU = tick_accumulator_VIU + iteration_work;

				stringstream_VIU << to_string(iteration) << " " << iteration_work << endl;
				stringstream_accum_VIU << to_string(iteration) << " " << tick_accumulator_VIU << endl;
		}

		//VIH testing
		printf("VIH\n");
		A_type A2 = copy_A(A);

		V_type V_heap_approx_tuple = value_iteration_with_heap(S, R, A2, P, gamma, epsilon);
		vector<double> V_heap_approx = get<0>(V_heap_approx_tuple);
		int V_heap_approx_iterations = get<1>(V_heap_approx_tuple);
		vector<microseconds> V_heap_approx_work_per_iteration = get<2>(V_heap_approx_tuple);

		auto tick_accumulator_VIH = V_heap_approx_work_per_iteration[0].count();
		
		for(int iteration = 1; iteration <= V_heap_approx_iterations; iteration++) {
				auto iteration_work = V_heap_approx_work_per_iteration[iteration].count();
				tick_accumulator_VIH = tick_accumulator_VIH + iteration_work;

				stringstream_VIH << to_string(iteration) << " " << iteration_work << endl;
				stringstream_accum_VIH << to_string(iteration) << " " << tick_accumulator_VIH << endl;
		}

		//BVI
		printf("BVI\n");
		A_type A3 = copy_A(A);

		V_type V_bounded_approx_solution_tuple = bounded_value_iteration(S, R, A3, P, gamma, epsilon);
		vector<double> V_bounded_approx_solution = get<0>(V_bounded_approx_solution_tuple);
		int V_bounded_approx_solution_iterations = get<1>(V_bounded_approx_solution_tuple);
		vector<microseconds> V_bounded_approx_solution_work_per_iteration = get<2>(V_bounded_approx_solution_tuple);

		auto tick_accumulator_BVI = V_bounded_approx_solution_work_per_iteration[0].count();
		
		for(int iteration = 1; iteration <= V_bounded_approx_solution_iterations; iteration++) {
				auto iteration_work = V_bounded_approx_solution_work_per_iteration[iteration].count();
				tick_accumulator_BVI = tick_accumulator_BVI + iteration_work;

				stringstream_BVI << to_string(iteration) << " " << iteration_work << endl;
				stringstream_accum_BVI << to_string(iteration) << " " << tick_accumulator_BVI << endl;
		}

		//VIAE
		printf("VIAE\n");
		A_type A4 = copy_A(A);

		V_type V_AE_approx_solution_tuple = value_iteration_action_elimination(S, R, A4, P, gamma, epsilon);
		vector<double> V_AE_approx_solution = get<0>(V_AE_approx_solution_tuple);
		int V_AE_approx_solution_iterations = get<1>(V_AE_approx_solution_tuple);
		vector<microseconds> V_AE_approx_solution_work_per_iteration = get<2>(V_AE_approx_solution_tuple);

		auto tick_accumulator_VIAE = V_AE_approx_solution_work_per_iteration[0].count();

		for(int iteration = 1; iteration <= V_AE_approx_solution_iterations; iteration++) {
				auto iteration_work = V_AE_approx_solution_work_per_iteration[iteration].count();
				tick_accumulator_VIAE = tick_accumulator_VIAE + iteration_work;

				stringstream_VIAE << to_string(iteration) << " " << iteration_work << endl;
				stringstream_accum_VIAE << to_string(iteration) << " " << tick_accumulator_VIAE << endl;
		}

		//VIAEHL
		printf("VIAEHL\n");
		A_type A7 = copy_A(A);

		V_type VIAEHL_approx_solution_tuple = value_iteration_action_elimination_heaps_lower_bound_approx(S, R, A7, P, gamma, epsilon);
		vector<double> VIAEHL_approx_solution = get<0>(VIAEHL_approx_solution_tuple);
		int VIAEHL_approx_solution_iterations = get<1>(VIAEHL_approx_solution_tuple);
		vector<microseconds> VIAEHL_approx_solution_work_per_iteration = get<2>(VIAEHL_approx_solution_tuple);

		auto tick_accumulator_VIAEHL = VIAEHL_approx_solution_work_per_iteration[0].count();

		for(int iteration = 1; iteration <= VIAEHL_approx_solution_iterations; iteration++) {
				auto iteration_work = VIAEHL_approx_solution_work_per_iteration[iteration].count();
				tick_accumulator_VIAEHL = tick_accumulator_VIAEHL + iteration_work;

				stringstream_VIAEHL << to_string(iteration) << " " << iteration_work << endl;
				stringstream_accum_VIAEHL << to_string(iteration) << " " << tick_accumulator_VIAEHL << endl;
		}

		//BAO
		printf("BAO\n");
		A_type A8 = copy_A(A);

		V_type BAO_approx_solution_tuple = value_iteration_BAO(S, R, A8, P, gamma, epsilon);
		vector<double> BAO_approx_solution = get<0>(BAO_approx_solution_tuple);
		int BAO_approx_solution_iterations = get<1>(BAO_approx_solution_tuple);
		vector<microseconds> BAO_approx_solution_work_per_iteration = get<2>(BAO_approx_solution_tuple);

		auto tick_accumulator_BAO = BAO_approx_solution_work_per_iteration[0].count();

		for(int iteration = 1; iteration <= BAO_approx_solution_iterations; iteration++) {
				auto iteration_work = BAO_approx_solution_work_per_iteration[iteration].count();
				tick_accumulator_BAO = tick_accumulator_BAO + iteration_work;

				stringstream_BAO << to_string(iteration) << " " << iteration_work << endl;
				stringstream_accum_BAO << to_string(iteration) << " " << tick_accumulator_BAO << endl;
		}

		//They should in theory all be epsilon from true value, and therefore, at most be 2 * epsilon from each other
		if (abs_max_diff_vectors(V_approx_solution, V_heap_approx) > (2 * epsilon)){
				printf("DIFFERENCE\n");
		}
		if (abs_max_diff_vectors(V_AE_approx_solution, V_AE_H_approx_solution) > (2 * epsilon)){
				printf("DIFFERENCE\n");
		}
		if (abs_max_diff_vectors(V_bounded_approx_solution, V_approx_solution_upper) > (2 * epsilon)){
				printf("DIFFERENCE\n");
		}
		if (abs_max_diff_vectors(VIAEHL_approx_solution, V_AE_H_approx_solution) > (2 * epsilon)){
				printf("DIFFERENCE\n");
		}
		if (abs_max_diff_vectors(BAO_approx_solution, V_AE_H_approx_solution) > (2 * epsilon)){
				printf("DIFFERENCE\n");
		}


		//WRITE ALL DATA TO THEIR RESPECTVIE FILES	
		write_stringstream_to_file(stringstream_VI, output_stream_VI, file_name_VI);
		write_stringstream_to_file(stringstream_VIU, output_stream_VIU, file_name_VIU);
		write_stringstream_to_file(stringstream_BVI, output_stream_BVI, file_name_BVI);
		write_stringstream_to_file(stringstream_VIH, output_stream_VIH, file_name_VIH);
		write_stringstream_to_file(stringstream_VIAE, output_stream_VIAE, file_name_VIAE);
		write_stringstream_to_file(stringstream_VIAEH, output_stream_VIAEH, file_name_VIAEH);
		write_stringstream_to_file(stringstream_VIAEHL, output_stream_VIAEHL, file_name_VIAEHL);
		write_stringstream_to_file(stringstream_BAO, output_stream_BAO, file_name_BAO);

		write_stringstream_to_file(stringstream_accum_VI, output_stream_accum_VI, file_name_accum_VI);
		write_stringstream_to_file(stringstream_accum_VIU, output_stream_accum_VIU, file_name_accum_VIU);
		write_stringstream_to_file(stringstream_accum_BVI, output_stream_accum_BVI, file_name_accum_BVI);
		write_stringstream_to_file(stringstream_accum_VIH, output_stream_accum_VIH, file_name_accum_VIH);
		write_stringstream_to_file(stringstream_accum_VIAE, output_stream_accum_VIAE, file_name_accum_VIAE);
		write_stringstream_to_file(stringstream_accum_VIAEH, output_stream_accum_VIAEH, file_name_accum_VIAEH);
		write_stringstream_to_file(stringstream_accum_VIAEHL, output_stream_accum_VIAEHL, file_name_accum_VIAEHL);
		write_stringstream_to_file(stringstream_accum_BAO, output_stream_accum_BAO, file_name_accum_BAO);
}

//TOP ACTION CHANGE EXPERIMENT
void write_meta_data_to_dat_file_top_action_change(ostringstream& string_stream, int S, int A_num, double epsilon, double gamma, int non_zero_transitions, double upper_reward, double action_prob){
		time_t time_now = time(0);
		string_stream << "# META DATA" << endl;
		string_stream << "# " << endl;
		string_stream << "# " << "experiment run at: " << ctime(&time_now);
		string_stream << "# " << endl;
		string_stream << "# " << "gamma = " << gamma << endl;
		string_stream << "# " << "epsilon = " << epsilon << endl;
		string_stream << "# " << "upper_reward = " << upper_reward << endl;
		string_stream << "# " << "S = " << S << endl;
		string_stream << "# " << "A = " << A_num << endl;
		string_stream << "# " << "non_zero_transitions = " << non_zero_transitions << endl;
		string_stream << "# " << "action_prob = " << action_prob << endl;
		string_stream << "# " << endl;
		string_stream << "# ACTUAL DATA" << endl;
		string_stream << "# iteration number | top action changes in current iteration" << endl;
}

void create_data_tables_top_action_change(string filename, int S, int A_num, double epsilon, double gamma, double upper_reward, double action_prob, int number_of_transitions, double mean, double variance, double lambda){

		//FOR WORK PER ITERATION
		//the stringstreams to create the test for the files
		ostringstream stringstream_uniform;
		ostringstream stringstream_normal;
		ostringstream stringstream_exponential;

		//the file output objects
		ofstream output_stream_uniform;
		ofstream output_stream_normal;
		ofstream output_stream_exponential;
		
		//set the name of the file to write to
		string file_name_uniform = "data_tables/top_action_change/" + filename + "_uniform.dat";
		string file_name_normal = "data_tables/top_action_change/" + filename + "_normal.dat";
		string file_name_exponential = "data_tables/top_action_change/" + filename + "_exponential.dat";

		//write meta data to all stringstreams as first in their respective files
		write_meta_data_to_dat_file_top_action_change(stringstream_uniform, S, A_num, epsilon, gamma, number_of_transitions, upper_reward, action_prob);
		write_meta_data_to_dat_file_top_action_change(stringstream_normal, S, A_num, epsilon, gamma, number_of_transitions, upper_reward, action_prob);
		write_meta_data_to_dat_file_top_action_change(stringstream_exponential, S, A_num, epsilon, gamma, number_of_transitions, upper_reward, action_prob);
		
		//UNIFORM REWARDS
		int seed = time(0);
		auto MDP_uniform_reward = generate_random_MDP_with_variable_parameters_fixed_nonzero_trans_states(S, A_num, action_prob, number_of_transitions, upper_reward, seed);
		R_type R_uniform_reward = get<0>(MDP_uniform_reward);
		A_type A_uniform_reward = get<1>(MDP_uniform_reward);
		P_type P_uniform_reward = get<2>(MDP_uniform_reward);

		printf("Beginning: uniform\n");
		A_type A1 = copy_A(A_uniform_reward);

		tuple<int, vector<int>> top_action_change_uniform_reward_tuple = top_action_change(S, R_uniform_reward, A1, P_uniform_reward, gamma, epsilon);
		int total_top_action_changes_uniform_reward = get<0>(top_action_change_uniform_reward_tuple);
		vector<int> top_action_change_per_iteration_uniform_reward = get<1>(top_action_change_uniform_reward_tuple);

		// -1 as there is the 0th element as dummy
		int number_of_iterations_uniform_reward = int(top_action_change_per_iteration_uniform_reward.size()) - 1;
		
		for(int iteration = 1; iteration <= number_of_iterations_uniform_reward; iteration++) {
				stringstream_uniform << to_string(iteration) << " " << (float(top_action_change_per_iteration_uniform_reward[iteration]) / float(S)) << endl;
		}

		//NORMAL DISTRIBUTED REWARDS
		seed = time(0);
		auto MDP_normal_reward = generate_random_MDP_normal_distributed_rewards(S, A_num, action_prob, number_of_transitions, seed, mean, variance);
		R_type R_normal_reward = get<0>(MDP_normal_reward);
		A_type A_normal_reward = get<1>(MDP_normal_reward);
		P_type P_normal_reward = get<2>(MDP_normal_reward);

		printf("Beginning: Normal\n");
		A_type A2 = copy_A(A_normal_reward);

		tuple<int, vector<int>> top_action_change_normal_reward_tuple = top_action_change(S, R_normal_reward, A2, P_normal_reward, gamma, epsilon);
		int total_top_action_changes_normal_reward = get<0>(top_action_change_normal_reward_tuple);
		vector<int> top_action_change_per_iteration_normal_reward = get<1>(top_action_change_normal_reward_tuple);

		// -1 as there is the 0th element as dummy
		int number_of_iterations_normal_reward = int(top_action_change_per_iteration_normal_reward.size()) - 1;
		
		for(int iteration = 1; iteration <= number_of_iterations_normal_reward; iteration++) {
				stringstream_normal << to_string(iteration) << " " << (float(top_action_change_per_iteration_normal_reward[iteration]) / float(S)) << endl;
		}

		//EXPONENTIAL DISTRIBUTED REWARDS
		seed = time(0);
		auto MDP_exponential_reward = generate_random_MDP_exponential_distributed_rewards(S, A_num, action_prob, number_of_transitions, lambda, seed);
		R_type R_exponential_reward = get<0>(MDP_exponential_reward);
		A_type A_exponential_reward = get<1>(MDP_exponential_reward);
		P_type P_exponential_reward = get<2>(MDP_exponential_reward);

		printf("Beginning: Exponential\n");
		A_type A3 = copy_A(A_exponential_reward);

		tuple<int, vector<int>> top_action_change_exponential_reward_tuple = top_action_change(S, R_exponential_reward, A3, P_exponential_reward, gamma, epsilon);
		int total_top_action_changes_exponential_reward = get<0>(top_action_change_exponential_reward_tuple);
		vector<int> top_action_change_per_iteration_exponential_reward = get<1>(top_action_change_exponential_reward_tuple);

		// -1 as there is the 0th element as dummy
		int number_of_iterations_exponential_reward = int(top_action_change_per_iteration_exponential_reward.size()) - 1;
		
		for(int iteration = 1; iteration <= number_of_iterations_exponential_reward; iteration++) {
				stringstream_exponential << to_string(iteration) << " " << (float(top_action_change_per_iteration_exponential_reward[iteration]) / float(S)) << endl;
		}

		//WRITE ALL DATA TO THEIR RESPECTVIE FILES	
		write_stringstream_to_file(stringstream_normal, output_stream_normal, file_name_normal);
		write_stringstream_to_file(stringstream_uniform, output_stream_uniform, file_name_uniform);
		write_stringstream_to_file(stringstream_exponential, output_stream_exponential, file_name_exponential);
}

void write_meta_data_to_dat_file_normal_dist_varying_variance(ostringstream& string_stream, int S, int A_num, double epsilon, double gamma, int transition_list_size, double action_prob, double mean, double min_variance, double max_variance, double variance_increment){
		time_t time_now = time(0);
		string_stream << "# META DATA" << endl;
		string_stream << "# " << endl;
		string_stream << "# " << "experiment run at: " << ctime(&time_now);
		string_stream << "# " << endl;
		string_stream << "# " << "gamma = " << gamma << endl;
		string_stream << "# " << "epsilon = " << epsilon << endl;
		string_stream << "# " << "S = " << S << endl;
		string_stream << "# " << "A = " << A_num << endl;
		string_stream << "# " << "transition list size in each state = " << transition_list_size << endl;
		string_stream << "# " << "action_prob = " << action_prob << endl;
		string_stream << "# " << "mean = " << mean << endl;
		string_stream << "# " << endl;
		string_stream << "# VARIABLE IN THIS EXPERIMENT" << endl;
		string_stream << "# " << "variance is varied from " << min_variance << " to " << max_variance << " with " << variance_increment << " increment" << endl;
		string_stream << "# " << endl;
		string_stream << "# ACTUAL DATA" << endl;
		string_stream << "# variance | microseconds" << endl;
}

void create_data_tables_normal_dist_varying_variance(string filename, int S, int A_num, double epsilon, double gamma, double action_prob, int transition_list_size, double mean, double min_variance, double max_variance, double variance_increment){

		//the stringstreams to create the test for the files
		ostringstream stringstream_BVI;
		ostringstream stringstream_VI;
		ostringstream stringstream_VIU;
		ostringstream stringstream_VIH;
		ostringstream stringstream_VIAE;
		ostringstream stringstream_VIAEH;

		//the file output objects
		ofstream output_stream_BVI;
		ofstream output_stream_VI;
		ofstream output_stream_VIU;
		ofstream output_stream_VIH;
		ofstream output_stream_VIAE;
		ofstream output_stream_VIAEH;
		
		//set the name of the file to write to
		string file_name_BVI = "data_tables/normal_dist_varying_variance/" + filename + "_BVI.dat"; 
		string file_name_VI =  "data_tables/normal_dist_varying_variance/" + filename + "_VI.dat";
		string file_name_VIU = "data_tables/normal_dist_varying_variance/" + filename + "_VIU.dat";
		string file_name_VIH = "data_tables/normal_dist_varying_variance/" + filename + "_VIH.dat";
		string file_name_VIAE = "data_tables/normal_dist_varying_variance/" + filename + "_VIAE.dat";
		string file_name_VIAEH = "data_tables/normal_dist_varying_variance/" + filename + "_VIAEH.dat";

		//write meta data to all stringstreams as first in their respective files
		write_meta_data_to_dat_file_normal_dist_varying_variance(stringstream_VI, S, A_num, epsilon, gamma, transition_list_size, action_prob, mean, min_variance, max_variance, variance_increment);
		write_meta_data_to_dat_file_normal_dist_varying_variance(stringstream_VIU, S, A_num, epsilon, gamma, transition_list_size, action_prob, mean, min_variance, max_variance, variance_increment);
		write_meta_data_to_dat_file_normal_dist_varying_variance(stringstream_VIH, S, A_num, epsilon, gamma, transition_list_size, action_prob, mean, min_variance, max_variance, variance_increment);
		write_meta_data_to_dat_file_normal_dist_varying_variance(stringstream_BVI, S, A_num, epsilon, gamma, transition_list_size, action_prob, mean, min_variance, max_variance, variance_increment);
		write_meta_data_to_dat_file_normal_dist_varying_variance(stringstream_VIAE, S, A_num, epsilon, gamma, transition_list_size, action_prob, mean, min_variance, max_variance, variance_increment);
		write_meta_data_to_dat_file_normal_dist_varying_variance(stringstream_VIAEH, S, A_num, epsilon, gamma, transition_list_size, action_prob, mean, min_variance, max_variance, variance_increment);

		for (double variance = min_variance; variance <= max_variance; variance += variance_increment){
				
				//status message of experiment
				printf("\nBeginning iteration variance = %f\n", variance);

				//GENERATE THE MDP FROM CURRENT TIME SEED
				int seed = time(0);
				auto MDP = generate_random_MDP_normal_distributed_rewards(S, A_num, action_prob, transition_list_size, seed, mean, variance);
				R_type R = get<0>(MDP);
				A_type A = get<1>(MDP);
				P_type P = get<2>(MDP);

				//iterations test printing
				float R_max = find_max_R(R);
				float R_min = find_min_R(R);
				float iterations_bound = log(R_max /((1.0 - gamma) * epsilon)) / log(1.0 / gamma);
				printf("R_max is %f\n", R_max);
				printf("R_min is %f\n", R_min);
				printf("upper bound is %f\n", R_max / (1.0 - gamma));
				printf("lower bound is %f\n", R_min / (1.0 - gamma));
				printf("iterations bound: %f\n", iterations_bound);
				

				//VI testing
				A_type A1 = copy_A(A);
				auto start_VI = high_resolution_clock::now();

				V_type V_approx_solution_tuple = value_iteration(S, R, A1, P, gamma, epsilon);
                vector<double> V_approx_solution = get<0>(V_approx_solution_tuple);

				auto stop_VI = high_resolution_clock::now();
				auto duration_VI = duration_cast<microseconds>(stop_VI - start_VI);

				stringstream_VI << to_string(variance) << " " << duration_VI.count() << endl;

				//VIU testing
				A_type A6 = copy_A(A);
				auto start_VIU = high_resolution_clock::now();

				V_type V_approx_solution_upper_tuple = value_iteration_upper(S, R, A6, P, gamma, epsilon);
				vector<double> V_approx_solution_upper = get<0>(V_approx_solution_upper_tuple);

				auto stop_VIU = high_resolution_clock::now();
				auto duration_VIU = duration_cast<microseconds>(stop_VIU - start_VIU);

				stringstream_VIU << to_string(variance) << " " << duration_VIU.count() << endl;

				//VIH testing
				A_type A2 = copy_A(A);
				auto start_VIH = high_resolution_clock::now();

				V_type V_heap_approx_tuple = value_iteration_with_heap(S, R, A2, P, gamma, epsilon);
				vector<double> V_heap_approx = get<0>(V_heap_approx_tuple);

				auto stop_VIH = high_resolution_clock::now();
				auto duration_VIH = duration_cast<microseconds>(stop_VIH - start_VIH);

				stringstream_VIH << to_string(variance) << " " << duration_VIH.count() << endl;

				//BVI
				A_type A3 = copy_A(A);
				auto start_BVI = high_resolution_clock::now();

				V_type V_bounded_approx_solution_tuple = bounded_value_iteration(S, R, A3, P, gamma, epsilon);
				vector<double> V_bounded_approx_solution = get<0>(V_bounded_approx_solution_tuple);

				auto stop_BVI = high_resolution_clock::now();
				auto duration_BVI = duration_cast<microseconds>(stop_BVI - start_BVI);
				
				stringstream_BVI << to_string(variance) << " " << duration_BVI.count() << endl;

				//VIAE
				A_type A4 = copy_A(A);
				auto start_VIAE = high_resolution_clock::now();

				V_type V_AE_approx_solution_tuple = value_iteration_action_elimination(S, R, A4, P, gamma, epsilon);
				vector<double> V_AE_approx_solution = get<0>(V_AE_approx_solution_tuple);

				auto stop_VIAE = high_resolution_clock::now();
				auto duration_VIAE = duration_cast<microseconds>(stop_VIAE - start_VIAE);

				stringstream_VIAE << to_string(variance) << " " << duration_VIAE.count() << endl;

				//VIAEH
				A_type A5 = copy_A(A);
				auto start_VIAEH = high_resolution_clock::now();

				V_type V_AE_H_approx_solution_tuple = value_iteration_action_elimination_heaps(S, R, A5, P, gamma, epsilon);
				vector<double> V_AE_H_approx_solution = get<0>(V_AE_H_approx_solution_tuple);

				auto stop_VIAEH = high_resolution_clock::now();
				auto duration_VIAEH = duration_cast<microseconds>(stop_VIAEH - start_VIAEH);

				stringstream_VIAEH << to_string(variance) << " " << duration_VIAEH.count() << endl;

				//They should in theory all be epsilon from true value, and therefore, at most be 2 * epsilon from each other
				if (abs_max_diff_vectors(V_approx_solution, V_heap_approx) > (2 * epsilon)){
						printf("DIFFERENCE\n");
				}
				if (abs_max_diff_vectors(V_AE_approx_solution, V_AE_H_approx_solution) > (2 * epsilon)){
						printf("DIFFERENCE\n");
				}
				if (abs_max_diff_vectors(V_bounded_approx_solution, V_approx_solution_upper) > (2 * epsilon)){
						printf("DIFFERENCE\n");
				}

		}

		//WRITE ALL DATA TO THEIR RESPECTVIE FILES	
		write_stringstream_to_file(stringstream_VI, output_stream_VI, file_name_VI);
		write_stringstream_to_file(stringstream_VIU, output_stream_VIU, file_name_VIU);
		write_stringstream_to_file(stringstream_BVI, output_stream_BVI, file_name_BVI);
		write_stringstream_to_file(stringstream_VIH, output_stream_VIH, file_name_VIH);
		write_stringstream_to_file(stringstream_VIAE, output_stream_VIAE, file_name_VIAE);
		write_stringstream_to_file(stringstream_VIAEH, output_stream_VIAEH, file_name_VIAEH);

}

//VIH DIFFERENT REWARD DISTRIBUTIONS - WORK PER ITERATION
void write_meta_data_to_dat_file_VIH_distributions_iterations_work(ostringstream& string_stream, int S, int A_num, double epsilon, double gamma, int non_zero_transitions, double upper_reward, double action_prob){
		time_t time_now = time(0);
		string_stream << "# META DATA" << endl;
		string_stream << "# " << endl;
		string_stream << "# " << "experiment run at: " << ctime(&time_now);
		string_stream << "# " << endl;
		string_stream << "# " << "gamma = " << gamma << endl;
		string_stream << "# " << "epsilon = " << epsilon << endl;
		string_stream << "# " << "upper_reward = " << upper_reward << endl;
		string_stream << "# " << "S = " << S << endl;
		string_stream << "# " << "A = " << A_num << endl;
		string_stream << "# " << "non_zero_transitions = " << non_zero_transitions << endl;
		string_stream << "# " << "action_prob = " << action_prob << endl;
		string_stream << "# " << endl;
		string_stream << "# ACTUAL DATA" << endl;
		string_stream << "# iteration number | top action changes in current iteration" << endl;
}

void create_data_tables_VIH_distributions_iterations_work(string filename, int S, int A_num, double epsilon, double gamma, double action_prob, int number_of_transitions, double mean_1, double mean_2, double mean_3, double variance_1, double variance_2, double variance_3, double lambda_1, double lambda_2, double lambda_3, double upper_reward_1, double upper_reward_2, double upper_reward_3){

		//FOR WORK PER ITERATION
		//the stringstreams to create the test for the files
		ostringstream stringstream_uniform_1;
		ostringstream stringstream_uniform_2;
		ostringstream stringstream_uniform_3;
		ostringstream stringstream_normal_1;
		ostringstream stringstream_normal_2;
		ostringstream stringstream_normal_3;
		ostringstream stringstream_exponential_1;
		ostringstream stringstream_exponential_2;
		ostringstream stringstream_exponential_3;

		//the file output objects
		ofstream output_stream_uniform_1;
		ofstream output_stream_uniform_2;
		ofstream output_stream_uniform_3;
		ofstream output_stream_normal_1;
		ofstream output_stream_normal_2;
		ofstream output_stream_normal_3;
		ofstream output_stream_exponential_1;
		ofstream output_stream_exponential_2;
		ofstream output_stream_exponential_3;
		
		//set the name of the file to write to
		string file_name_uniform_1 = "data_tables/VIH_distributions_iterations_work/" + filename + "_uniform_1.dat";
		string file_name_uniform_2 = "data_tables/VIH_distributions_iterations_work/" + filename + "_uniform_2.dat";
		string file_name_uniform_3 = "data_tables/VIH_distributions_iterations_work/" + filename + "_uniform_3.dat";
		string file_name_normal_1 = "data_tables/VIH_distributions_iterations_work/" + filename + "_normal_1.dat";
		string file_name_normal_2 = "data_tables/VIH_distributions_iterations_work/" + filename + "_normal_2.dat";
		string file_name_normal_3 = "data_tables/VIH_distributions_iterations_work/" + filename + "_normal_3.dat";
		string file_name_exponential_1 = "data_tables/VIH_distributions_iterations_work/" + filename + "_exponential_1.dat";
		string file_name_exponential_2 = "data_tables/VIH_distributions_iterations_work/" + filename + "_exponential_2.dat";
		string file_name_exponential_3 = "data_tables/VIH_distributions_iterations_work/" + filename + "_exponential_3.dat";

		//write meta data to all stringstreams as first in their respective files
		//write_meta_data_to_dat_file_work_per_iteration(stringstream_uniform_1, S, A_num, epsilon, gamma, number_of_transitions, upper_reward, action_prob);
		//write_meta_data_to_dat_file_work_per_iteration(stringstream_uniform_2, S, A_num, epsilon, gamma, number_of_transitions, upper_reward, action_prob);
		//write_meta_data_to_dat_file_work_per_iteration(stringstream_uniform_3, S, A_num, epsilon, gamma, number_of_transitions, upper_reward, action_prob);
		//write_meta_data_to_dat_file_work_per_iteration(stringstream_normal_1, S, A_num, epsilon, gamma, number_of_transitions, upper_reward, action_prob);
		//write_meta_data_to_dat_file_work_per_iteration(stringstream_normal_2, S, A_num, epsilon, gamma, number_of_transitions, upper_reward, action_prob);
		//write_meta_data_to_dat_file_work_per_iteration(stringstream_normal_3, S, A_num, epsilon, gamma, number_of_transitions, upper_reward, action_prob);
		//write_meta_data_to_dat_file_work_per_iteration(stringstream_exponential_1, S, A_num, epsilon, gamma, number_of_transitions, upper_reward, action_prob);
		//write_meta_data_to_dat_file_work_per_iteration(stringstream_exponential_2, S, A_num, epsilon, gamma, number_of_transitions, upper_reward, action_prob);
		//write_meta_data_to_dat_file_work_per_iteration(stringstream_exponential_3, S, A_num, epsilon, gamma, number_of_transitions, upper_reward, action_prob);
	
		//ACCUMULATED REWARDS
		//FOR WORK PER ITERATION
		//the stringstreams to create the test for the files
		ostringstream stringstream_uniform_acc_1;
		ostringstream stringstream_uniform_acc_2;
		ostringstream stringstream_uniform_acc_3;
		ostringstream stringstream_normal_acc_1;
		ostringstream stringstream_normal_acc_2;
		ostringstream stringstream_normal_acc_3;
		ostringstream stringstream_exponential_acc_1;
		ostringstream stringstream_exponential_acc_2;
		ostringstream stringstream_exponential_acc_3;

		//the file output objects
		ofstream output_stream_uniform_acc_1;
		ofstream output_stream_uniform_acc_2;
		ofstream output_stream_uniform_acc_3;
		ofstream output_stream_normal_acc_1;
		ofstream output_stream_normal_acc_2;
		ofstream output_stream_normal_acc_3;
		ofstream output_stream_exponential_acc_1;
		ofstream output_stream_exponential_acc_2;
		ofstream output_stream_exponential_acc_3;
		
		//set the name of the file to write to
		string file_name_uniform_acc_1 = "data_tables/VIH_distributions_iterations_work/" + filename + "_uniform_acc_1.dat";
		string file_name_uniform_acc_2 = "data_tables/VIH_distributions_iterations_work/" + filename + "_uniform_acc_2.dat";
		string file_name_uniform_acc_3 = "data_tables/VIH_distributions_iterations_work/" + filename + "_uniform_acc_3.dat";
		string file_name_normal_acc_1 = "data_tables/VIH_distributions_iterations_work/" + filename + "_normal_acc_1.dat";
		string file_name_normal_acc_2 = "data_tables/VIH_distributions_iterations_work/" + filename + "_normal_acc_2.dat";
		string file_name_normal_acc_3 = "data_tables/VIH_distributions_iterations_work/" + filename + "_normal_acc_3.dat";
		string file_name_exponential_acc_1 = "data_tables/VIH_distributions_iterations_work/" + filename + "_exponential_acc_1.dat";
		string file_name_exponential_acc_2 = "data_tables/VIH_distributions_iterations_work/" + filename + "_exponential_acc_2.dat";
		string file_name_exponential_acc_3 = "data_tables/VIH_distributions_iterations_work/" + filename + "_exponential_acc_3.dat";

		//write meta data to all stringstreams as first in their respective files
		//write_meta_data_to_dat_file_work_per_iteration(stringstream_uniform_acc_1, S, A_num, epsilon, gamma, number_of_transitions, upper_reward, action_prob);
		//write_meta_data_to_dat_file_work_per_iteration(stringstream_uniform_acc_2, S, A_num, epsilon, gamma, number_of_transitions, upper_reward, action_prob);
		//write_meta_data_to_dat_file_work_per_iteration(stringstream_uniform_acc_3, S, A_num, epsilon, gamma, number_of_transitions, upper_reward, action_prob);
		//write_meta_data_to_dat_file_work_per_iteration(stringstream_normal_acc_1, S, A_num, epsilon, gamma, number_of_transitions, upper_reward, action_prob);
		//write_meta_data_to_dat_file_work_per_iteration(stringstream_normal_acc_2, S, A_num, epsilon, gamma, number_of_transitions, upper_reward, action_prob);
		//write_meta_data_to_dat_file_work_per_iteration(stringstream_normal_acc_3, S, A_num, epsilon, gamma, number_of_transitions, upper_reward, action_prob);
		//write_meta_data_to_dat_file_work_per_iteration(stringstream_exponential_acc_1, S, A_num, epsilon, gamma, number_of_transitions, upper_reward, action_prob);
		//write_meta_data_to_dat_file_work_per_iteration(stringstream_exponential_acc_2, S, A_num, epsilon, gamma, number_of_transitions, upper_reward, action_prob);
		//write_meta_data_to_dat_file_work_per_iteration(stringstream_exponential_acc_3, S, A_num, epsilon, gamma, number_of_transitions, upper_reward, action_prob);

		//UNIFORM REWARDS 1
		int seed = time(0);
		auto MDP_uniform_1_reward = generate_random_MDP_with_variable_parameters_fixed_nonzero_trans_states(S, A_num, action_prob, number_of_transitions, upper_reward_1, seed);
		R_type R_uniform_1_reward = get<0>(MDP_uniform_1_reward);
		A_type A_uniform_1_reward = get<1>(MDP_uniform_1_reward);
		P_type P_uniform_1_reward = get<2>(MDP_uniform_1_reward);

		printf("Beginning: uniform_1\n");
		A_type A1 = copy_A(A_uniform_1_reward);

		V_type V_heap_approx_tuple_uniform_1 = value_iteration_with_heap(S, R_uniform_1_reward, A1, P_uniform_1_reward, gamma, epsilon);
		vector<double> V_heap_approx_uniform_1 = get<0>(V_heap_approx_tuple_uniform_1);
		int V_heap_approx_iterations_uniform_1 = get<1>(V_heap_approx_tuple_uniform_1);
		vector<microseconds> V_heap_approx_work_per_iteration_uniform_1 = get<2>(V_heap_approx_tuple_uniform_1);

		auto tick_accumulator_VIH_uniform_1 = V_heap_approx_work_per_iteration_uniform_1[0].count();
		
		for(int iteration = 1; iteration <= V_heap_approx_iterations_uniform_1; iteration++) {
				auto iteration_work_uniform_1 = V_heap_approx_work_per_iteration_uniform_1[iteration].count();
				tick_accumulator_VIH_uniform_1 = tick_accumulator_VIH_uniform_1 + iteration_work_uniform_1;

				stringstream_uniform_1 << to_string(iteration) << " " << iteration_work_uniform_1 << endl;
				stringstream_uniform_acc_1 << to_string(iteration) << " " << tick_accumulator_VIH_uniform_1 << endl;
		}

		//UNIFORM REWARDS 2
		seed = time(0);
		auto MDP_uniform_2_reward = generate_random_MDP_with_variable_parameters_fixed_nonzero_trans_states(S, A_num, action_prob, number_of_transitions, upper_reward_2, seed);
		R_type R_uniform_2_reward = get<0>(MDP_uniform_2_reward);
		A_type A_uniform_2_reward = get<1>(MDP_uniform_2_reward);
		P_type P_uniform_2_reward = get<2>(MDP_uniform_2_reward);

		printf("Beginning: uniform_2\n");
		A_type A2 = copy_A(A_uniform_2_reward);

		V_type V_heap_approx_tuple_uniform_2 = value_iteration_with_heap(S, R_uniform_2_reward, A2, P_uniform_2_reward, gamma, epsilon);
		vector<double> V_heap_approx_uniform_2 = get<0>(V_heap_approx_tuple_uniform_2);
		int V_heap_approx_iterations_uniform_2 = get<1>(V_heap_approx_tuple_uniform_2);
		vector<microseconds> V_heap_approx_work_per_iteration_uniform_2 = get<2>(V_heap_approx_tuple_uniform_2);

		auto tick_accumulator_VIH_uniform_2 = V_heap_approx_work_per_iteration_uniform_2[0].count();
		
		for(int iteration = 1; iteration <= V_heap_approx_iterations_uniform_2; iteration++) {
				auto iteration_work_uniform_2 = V_heap_approx_work_per_iteration_uniform_2[iteration].count();
				tick_accumulator_VIH_uniform_2 = tick_accumulator_VIH_uniform_2 + iteration_work_uniform_2;

				stringstream_uniform_2 << to_string(iteration) << " " << iteration_work_uniform_2 << endl;
				stringstream_uniform_acc_2 << to_string(iteration) << " " << tick_accumulator_VIH_uniform_2 << endl;
		}

		//UNIFORM REWARDS 3
		seed = time(0);
		auto MDP_uniform_3_reward = generate_random_MDP_with_variable_parameters_fixed_nonzero_trans_states(S, A_num, action_prob, number_of_transitions, upper_reward_3, seed);
		R_type R_uniform_3_reward = get<0>(MDP_uniform_3_reward);
		A_type A_uniform_3_reward = get<1>(MDP_uniform_3_reward);
		P_type P_uniform_3_reward = get<2>(MDP_uniform_3_reward);

		printf("Beginning: uniform_3\n");
		A_type A3 = copy_A(A_uniform_3_reward);

		V_type V_heap_approx_tuple_uniform_3 = value_iteration_with_heap(S, R_uniform_3_reward, A3, P_uniform_3_reward, gamma, epsilon);
		vector<double> V_heap_approx_uniform_3 = get<0>(V_heap_approx_tuple_uniform_3);
		int V_heap_approx_iterations_uniform_3 = get<1>(V_heap_approx_tuple_uniform_3);
		vector<microseconds> V_heap_approx_work_per_iteration_uniform_3 = get<2>(V_heap_approx_tuple_uniform_3);

		auto tick_accumulator_VIH_uniform_3 = V_heap_approx_work_per_iteration_uniform_3[0].count();
		
		for(int iteration = 1; iteration <= V_heap_approx_iterations_uniform_3; iteration++) {
				auto iteration_work_uniform_3 = V_heap_approx_work_per_iteration_uniform_3[iteration].count();
				tick_accumulator_VIH_uniform_3 = tick_accumulator_VIH_uniform_3 + iteration_work_uniform_3;

				stringstream_uniform_3 << to_string(iteration) << " " << iteration_work_uniform_3 << endl;
				stringstream_uniform_acc_3 << to_string(iteration) << " " << tick_accumulator_VIH_uniform_3 << endl;
		}

		//NORMAL REWARDS 1
		seed = time(0);
		auto MDP_normal_1_reward = generate_random_MDP_normal_distributed_rewards(S, A_num, action_prob, number_of_transitions, seed, mean_1, variance_1);
		R_type R_normal_1_reward = get<0>(MDP_normal_1_reward);
		A_type A_normal_1_reward = get<1>(MDP_normal_1_reward);
		P_type P_normal_1_reward = get<2>(MDP_normal_1_reward);

		printf("Beginning: normal_1\n");
		A_type A4 = copy_A(A_normal_1_reward);

		V_type V_heap_approx_tuple_normal_1 = value_iteration_with_heap(S, R_normal_1_reward, A4, P_normal_1_reward, gamma, epsilon);
		vector<double> V_heap_approx_normal_1 = get<0>(V_heap_approx_tuple_normal_1);
		int V_heap_approx_iterations_normal_1 = get<1>(V_heap_approx_tuple_normal_1);
		vector<microseconds> V_heap_approx_work_per_iteration_normal_1 = get<2>(V_heap_approx_tuple_normal_1);

		auto tick_accumulator_VIH_normal_1 = V_heap_approx_work_per_iteration_normal_1[0].count();
		
		for(int iteration = 1; iteration <= V_heap_approx_iterations_normal_1; iteration++) {
				auto iteration_work_normal_1 = V_heap_approx_work_per_iteration_normal_1[iteration].count();
				tick_accumulator_VIH_normal_1 = tick_accumulator_VIH_normal_1 + iteration_work_normal_1;

				stringstream_normal_1 << to_string(iteration) << " " << iteration_work_normal_1 << endl;
				stringstream_normal_acc_1 << to_string(iteration) << " " << tick_accumulator_VIH_normal_1 << endl;
		}

		//NORMAL REWARDS 2
		seed = time(0);
		auto MDP_normal_2_reward = generate_random_MDP_normal_distributed_rewards(S, A_num, action_prob, number_of_transitions, seed, mean_2, variance_2);
		R_type R_normal_2_reward = get<0>(MDP_normal_2_reward);
		A_type A_normal_2_reward = get<1>(MDP_normal_2_reward);
		P_type P_normal_2_reward = get<2>(MDP_normal_2_reward);

		printf("Beginning: normal_2\n");
		A_type A5 = copy_A(A_normal_2_reward);

		V_type V_heap_approx_tuple_normal_2 = value_iteration_with_heap(S, R_normal_2_reward, A5, P_normal_2_reward, gamma, epsilon);
		vector<double> V_heap_approx_normal_2 = get<0>(V_heap_approx_tuple_normal_2);
		int V_heap_approx_iterations_normal_2 = get<1>(V_heap_approx_tuple_normal_2);
		vector<microseconds> V_heap_approx_work_per_iteration_normal_2 = get<2>(V_heap_approx_tuple_normal_2);

		auto tick_accumulator_VIH_normal_2 = V_heap_approx_work_per_iteration_normal_2[0].count();
		
		for(int iteration = 1; iteration <= V_heap_approx_iterations_normal_2; iteration++) {
				auto iteration_work_normal_2 = V_heap_approx_work_per_iteration_normal_2[iteration].count();
				tick_accumulator_VIH_normal_2 = tick_accumulator_VIH_normal_2 + iteration_work_normal_2;

				stringstream_normal_2 << to_string(iteration) << " " << iteration_work_normal_2 << endl;
				stringstream_normal_acc_2 << to_string(iteration) << " " << tick_accumulator_VIH_normal_2 << endl;
		}

		//NORMAL REWARDS 3
		seed = time(0);
		auto MDP_normal_3_reward = generate_random_MDP_normal_distributed_rewards(S, A_num, action_prob, number_of_transitions, seed, mean_3, variance_3);
		R_type R_normal_3_reward = get<0>(MDP_normal_3_reward);
		A_type A_normal_3_reward = get<1>(MDP_normal_3_reward);
		P_type P_normal_3_reward = get<2>(MDP_normal_3_reward);

		printf("Beginning: normal_3\n");
		A_type A6 = copy_A(A_normal_3_reward);

		V_type V_heap_approx_tuple_normal_3 = value_iteration_with_heap(S, R_normal_3_reward, A6, P_normal_3_reward, gamma, epsilon);
		vector<double> V_heap_approx_normal_3 = get<0>(V_heap_approx_tuple_normal_3);
		int V_heap_approx_iterations_normal_3 = get<1>(V_heap_approx_tuple_normal_3);
		vector<microseconds> V_heap_approx_work_per_iteration_normal_3 = get<2>(V_heap_approx_tuple_normal_3);

		auto tick_accumulator_VIH_normal_3 = V_heap_approx_work_per_iteration_normal_3[0].count();
		
		for(int iteration = 1; iteration <= V_heap_approx_iterations_normal_3; iteration++) {
				auto iteration_work_normal_3 = V_heap_approx_work_per_iteration_normal_3[iteration].count();
				tick_accumulator_VIH_normal_3 = tick_accumulator_VIH_normal_3 + iteration_work_normal_3;

				stringstream_normal_3 << to_string(iteration) << " " << iteration_work_normal_3 << endl;
				stringstream_normal_acc_3 << to_string(iteration) << " " << tick_accumulator_VIH_normal_3 << endl;
		}

		//EXPONENTIAL REWARDS 1
		seed = time(0);
		auto MDP_exponential_1_reward = generate_random_MDP_exponential_distributed_rewards(S, A_num, action_prob, number_of_transitions, lambda_1, seed);
		R_type R_exponential_1_reward = get<0>(MDP_exponential_1_reward);
		A_type A_exponential_1_reward = get<1>(MDP_exponential_1_reward);
		P_type P_exponential_1_reward = get<2>(MDP_exponential_1_reward);

		printf("Beginning: exponential_1\n");
		A_type A7 = copy_A(A_exponential_1_reward);

		V_type V_heap_approx_tuple_exponential_1 = value_iteration_with_heap(S, R_exponential_1_reward, A7, P_exponential_1_reward, gamma, epsilon);
		vector<double> V_heap_approx_exponential_1 = get<0>(V_heap_approx_tuple_exponential_1);
		int V_heap_approx_iterations_exponential_1 = get<1>(V_heap_approx_tuple_exponential_1);
		vector<microseconds> V_heap_approx_work_per_iteration_exponential_1 = get<2>(V_heap_approx_tuple_exponential_1);

		auto tick_accumulator_VIH_exponential_1 = V_heap_approx_work_per_iteration_exponential_1[0].count();
		
		for(int iteration = 1; iteration <= V_heap_approx_iterations_exponential_1; iteration++) {
				auto iteration_work_exponential_1 = V_heap_approx_work_per_iteration_exponential_1[iteration].count();
				tick_accumulator_VIH_exponential_1 = tick_accumulator_VIH_exponential_1 + iteration_work_exponential_1;

				stringstream_exponential_1 << to_string(iteration) << " " << iteration_work_exponential_1 << endl;
				stringstream_exponential_acc_1 << to_string(iteration) << " " << tick_accumulator_VIH_exponential_1 << endl;
		}

		//EXPONENTIAL REWARDS 2
		seed = time(0);
		auto MDP_exponential_2_reward = generate_random_MDP_exponential_distributed_rewards(S, A_num, action_prob, number_of_transitions, lambda_2, seed);
		R_type R_exponential_2_reward = get<0>(MDP_exponential_2_reward);
		A_type A_exponential_2_reward = get<1>(MDP_exponential_2_reward);
		P_type P_exponential_2_reward = get<2>(MDP_exponential_2_reward);

		printf("Beginning: exponential_2\n");
		A_type A8 = copy_A(A_exponential_2_reward);

		V_type V_heap_approx_tuple_exponential_2 = value_iteration_with_heap(S, R_exponential_2_reward, A8, P_exponential_2_reward, gamma, epsilon);
		vector<double> V_heap_approx_exponential_2 = get<0>(V_heap_approx_tuple_exponential_2);
		int V_heap_approx_iterations_exponential_2 = get<1>(V_heap_approx_tuple_exponential_2);
		vector<microseconds> V_heap_approx_work_per_iteration_exponential_2 = get<2>(V_heap_approx_tuple_exponential_2);

		auto tick_accumulator_VIH_exponential_2 = V_heap_approx_work_per_iteration_exponential_2[0].count();
		
		for(int iteration = 1; iteration <= V_heap_approx_iterations_exponential_2; iteration++) {
				auto iteration_work_exponential_2 = V_heap_approx_work_per_iteration_exponential_2[iteration].count();
				tick_accumulator_VIH_exponential_2 = tick_accumulator_VIH_exponential_2 + iteration_work_exponential_2;

				stringstream_exponential_2 << to_string(iteration) << " " << iteration_work_exponential_2 << endl;
				stringstream_exponential_acc_2 << to_string(iteration) << " " << tick_accumulator_VIH_exponential_2 << endl;
		}

		//EXPONENTIAL REWARDS 3
		seed = time(0);
		auto MDP_exponential_3_reward = generate_random_MDP_exponential_distributed_rewards(S, A_num, action_prob, number_of_transitions, lambda_3, seed);
		R_type R_exponential_3_reward = get<0>(MDP_exponential_3_reward);
		A_type A_exponential_3_reward = get<1>(MDP_exponential_3_reward);
		P_type P_exponential_3_reward = get<2>(MDP_exponential_3_reward);

		printf("Beginning: exponential_3\n");
		A_type A9 = copy_A(A_exponential_3_reward);

		V_type V_heap_approx_tuple_exponential_3 = value_iteration_with_heap(S, R_exponential_3_reward, A9, P_exponential_3_reward, gamma, epsilon);
		vector<double> V_heap_approx_exponential_3 = get<0>(V_heap_approx_tuple_exponential_3);
		int V_heap_approx_iterations_exponential_3 = get<1>(V_heap_approx_tuple_exponential_3);
		vector<microseconds> V_heap_approx_work_per_iteration_exponential_3 = get<2>(V_heap_approx_tuple_exponential_3);

		auto tick_accumulator_VIH_exponential_3 = V_heap_approx_work_per_iteration_exponential_3[0].count();
		
		for(int iteration = 1; iteration <= V_heap_approx_iterations_exponential_3; iteration++) {
				auto iteration_work_exponential_3 = V_heap_approx_work_per_iteration_exponential_3[iteration].count();
				tick_accumulator_VIH_exponential_3 = tick_accumulator_VIH_exponential_3 + iteration_work_exponential_3;

				stringstream_exponential_3 << to_string(iteration) << " " << iteration_work_exponential_3 << endl;
				stringstream_exponential_acc_3 << to_string(iteration) << " " << tick_accumulator_VIH_exponential_3 << endl;
		}

		//WRITE ALL DATA TO THEIR RESPECTVIE FILES	
		write_stringstream_to_file(stringstream_normal_1, output_stream_normal_1, file_name_normal_1);
		write_stringstream_to_file(stringstream_normal_2, output_stream_normal_2, file_name_normal_2);
		write_stringstream_to_file(stringstream_normal_3, output_stream_normal_3, file_name_normal_3);
		write_stringstream_to_file(stringstream_uniform_1, output_stream_uniform_1, file_name_uniform_1);
		write_stringstream_to_file(stringstream_uniform_2, output_stream_uniform_2, file_name_uniform_2);
		write_stringstream_to_file(stringstream_uniform_3, output_stream_uniform_3, file_name_uniform_3);
		write_stringstream_to_file(stringstream_exponential_1, output_stream_exponential_1, file_name_exponential_1);
		write_stringstream_to_file(stringstream_exponential_2, output_stream_exponential_2, file_name_exponential_2);
		write_stringstream_to_file(stringstream_exponential_3, output_stream_exponential_3, file_name_exponential_3);


		//WRITE ALL DATA TO THEIR RESPECTVIE FILES	
		write_stringstream_to_file(stringstream_normal_acc_1, output_stream_normal_acc_1, file_name_normal_acc_1);
		write_stringstream_to_file(stringstream_normal_acc_2, output_stream_normal_acc_2, file_name_normal_acc_2);
		write_stringstream_to_file(stringstream_normal_acc_3, output_stream_normal_acc_3, file_name_normal_acc_3);
		write_stringstream_to_file(stringstream_uniform_acc_1, output_stream_uniform_acc_1, file_name_uniform_acc_1);
		write_stringstream_to_file(stringstream_uniform_acc_2, output_stream_uniform_acc_2, file_name_uniform_acc_2);
		write_stringstream_to_file(stringstream_uniform_acc_3, output_stream_uniform_acc_3, file_name_uniform_acc_3);
		write_stringstream_to_file(stringstream_exponential_acc_1, output_stream_exponential_acc_1, file_name_exponential_acc_1);
		write_stringstream_to_file(stringstream_exponential_acc_2, output_stream_exponential_acc_2, file_name_exponential_acc_2);
		write_stringstream_to_file(stringstream_exponential_acc_3, output_stream_exponential_acc_3, file_name_exponential_acc_3);
}

//EXPONENTIAL EXPERIMENT - VARYING LAMBDA
void write_meta_data_to_dat_file_exponential_dist_varying_lambda(ostringstream& string_stream, int S, int A_num, double epsilon, double gamma, int transition_list_size, double action_prob, double min_lambda, double max_lambda, double lambda_increment){
		time_t time_now = time(0);
		string_stream << "# META DATA" << endl;
		string_stream << "# " << endl;
		string_stream << "# " << "experiment run at: " << ctime(&time_now);
		string_stream << "# " << endl;
		string_stream << "# " << "gamma = " << gamma << endl;
		string_stream << "# " << "epsilon = " << epsilon << endl;
		string_stream << "# " << "S = " << S << endl;
		string_stream << "# " << "A = " << A_num << endl;
		string_stream << "# " << "transition list size in each state = " << transition_list_size << endl;
		string_stream << "# " << "action_prob = " << action_prob << endl;
		string_stream << "# " << endl;
		string_stream << "# VARIABLE IN THIS EXPERIMENT" << endl;
		string_stream << "# " << "lambda is varied from " << min_lambda << " to " << max_lambda << " with " << lambda_increment << " increment" << endl;
		string_stream << "# " << endl;
		string_stream << "# ACTUAL DATA" << endl;
		string_stream << "# lambda | microseconds" << endl;
}

void create_data_tables_exponential_dist_varying_lambda(string filename, int S, int A_num, double epsilon, double gamma, double action_prob, int transition_list_size, double min_lambda, double max_lambda, double lambda_increment){

		//the stringstreams to create the test for the files
		ostringstream stringstream_BVI;
		ostringstream stringstream_VI;
		ostringstream stringstream_VIU;
		ostringstream stringstream_VIH;
		ostringstream stringstream_VIAE;
		ostringstream stringstream_VIAEH;

		//the file output objects
		ofstream output_stream_BVI;
		ofstream output_stream_VI;
		ofstream output_stream_VIU;
		ofstream output_stream_VIH;
		ofstream output_stream_VIAE;
		ofstream output_stream_VIAEH;
		
		//set the name of the file to write to
		string file_name_BVI = "data_tables/exponential_dist_varying_lambda/" + filename + "_BVI.dat"; 
		string file_name_VI =  "data_tables/exponential_dist_varying_lambda/" + filename + "_VI.dat";
		string file_name_VIU = "data_tables/exponential_dist_varying_lambda/" + filename + "_VIU.dat";
		string file_name_VIH = "data_tables/exponential_dist_varying_lambda/" + filename + "_VIH.dat";
		string file_name_VIAE = "data_tables/exponential_dist_varying_lambda/" + filename + "_VIAE.dat";
		string file_name_VIAEH = "data_tables/exponential_dist_varying_lambda/" + filename + "_VIAEH.dat";

		//write meta data to all stringstreams as first in their respective files
		write_meta_data_to_dat_file_exponential_dist_varying_lambda(stringstream_VI, S, A_num, epsilon, gamma, transition_list_size, action_prob, min_lambda, max_lambda, lambda_increment);
		write_meta_data_to_dat_file_exponential_dist_varying_lambda(stringstream_VIU, S, A_num, epsilon, gamma, transition_list_size, action_prob, min_lambda, max_lambda, lambda_increment);
		write_meta_data_to_dat_file_exponential_dist_varying_lambda(stringstream_VIH, S, A_num, epsilon, gamma, transition_list_size, action_prob, min_lambda, max_lambda, lambda_increment);
		write_meta_data_to_dat_file_exponential_dist_varying_lambda(stringstream_BVI, S, A_num, epsilon, gamma, transition_list_size, action_prob, min_lambda, max_lambda, lambda_increment);
		write_meta_data_to_dat_file_exponential_dist_varying_lambda(stringstream_VIAE, S, A_num, epsilon, gamma, transition_list_size, action_prob, min_lambda, max_lambda, lambda_increment);
		write_meta_data_to_dat_file_exponential_dist_varying_lambda(stringstream_VIAEH, S, A_num, epsilon, gamma, transition_list_size, action_prob, min_lambda, max_lambda, lambda_increment);

		for (double lambda = min_lambda; lambda <= max_lambda; lambda += lambda_increment){
				
				//status message of experiment
				printf("\nBeginning iteration lambda = %f\n", lambda);

				//GENERATE THE MDP FROM CURRENT TIME SEED
				int seed = time(0);
				auto MDP = generate_random_MDP_exponential_distributed_rewards(S, A_num, action_prob, transition_list_size, lambda, seed);
				R_type R = get<0>(MDP);
				A_type A = get<1>(MDP);
				P_type P = get<2>(MDP);

				//iterations test printing
				float R_max = find_max_R(R);
				float R_min = find_min_R(R);
				float iterations_bound = log(R_max /((1.0 - gamma) * epsilon)) / log(1.0 / gamma);
				printf("R_max is %f\n", R_max);
				printf("R_min is %f\n", R_min);
				printf("upper bound is %f\n", R_max / (1.0 - gamma));
				printf("lower bound is %f\n", R_min / (1.0 - gamma));
				printf("iterations bound: %f\n", iterations_bound);
				

				//VI testing
				A_type A1 = copy_A(A);
				auto start_VI = high_resolution_clock::now();

				V_type V_approx_solution_tuple = value_iteration(S, R, A1, P, gamma, epsilon);
                vector<double> V_approx_solution = get<0>(V_approx_solution_tuple);

				auto stop_VI = high_resolution_clock::now();
				auto duration_VI = duration_cast<microseconds>(stop_VI - start_VI);

				stringstream_VI << to_string(lambda) << " " << duration_VI.count() << endl;

				//VIU testing
				A_type A6 = copy_A(A);
				auto start_VIU = high_resolution_clock::now();

				V_type V_approx_solution_upper_tuple = value_iteration_upper(S, R, A6, P, gamma, epsilon);
				vector<double> V_approx_solution_upper = get<0>(V_approx_solution_upper_tuple);

				auto stop_VIU = high_resolution_clock::now();
				auto duration_VIU = duration_cast<microseconds>(stop_VIU - start_VIU);

				stringstream_VIU << to_string(lambda) << " " << duration_VIU.count() << endl;

				//VIH testing
				A_type A2 = copy_A(A);
				auto start_VIH = high_resolution_clock::now();

				V_type V_heap_approx_tuple = value_iteration_with_heap(S, R, A2, P, gamma, epsilon);
				vector<double> V_heap_approx = get<0>(V_heap_approx_tuple);

				auto stop_VIH = high_resolution_clock::now();
				auto duration_VIH = duration_cast<microseconds>(stop_VIH - start_VIH);

				stringstream_VIH << to_string(lambda) << " " << duration_VIH.count() << endl;

				//BVI
				A_type A3 = copy_A(A);
				auto start_BVI = high_resolution_clock::now();

				V_type V_bounded_approx_solution_tuple = bounded_value_iteration(S, R, A3, P, gamma, epsilon);
				vector<double> V_bounded_approx_solution = get<0>(V_bounded_approx_solution_tuple);

				auto stop_BVI = high_resolution_clock::now();
				auto duration_BVI = duration_cast<microseconds>(stop_BVI - start_BVI);
				
				stringstream_BVI << to_string(lambda) << " " << duration_BVI.count() << endl;

				//VIAE
				A_type A4 = copy_A(A);
				auto start_VIAE = high_resolution_clock::now();

				V_type V_AE_approx_solution_tuple = value_iteration_action_elimination(S, R, A4, P, gamma, epsilon);
				vector<double> V_AE_approx_solution = get<0>(V_AE_approx_solution_tuple);

				auto stop_VIAE = high_resolution_clock::now();
				auto duration_VIAE = duration_cast<microseconds>(stop_VIAE - start_VIAE);

				stringstream_VIAE << to_string(lambda) << " " << duration_VIAE.count() << endl;

				//VIAEH
				A_type A5 = copy_A(A);
				auto start_VIAEH = high_resolution_clock::now();

				V_type V_AE_H_approx_solution_tuple = value_iteration_action_elimination_heaps(S, R, A5, P, gamma, epsilon);
				vector<double> V_AE_H_approx_solution = get<0>(V_AE_H_approx_solution_tuple);

				auto stop_VIAEH = high_resolution_clock::now();
				auto duration_VIAEH = duration_cast<microseconds>(stop_VIAEH - start_VIAEH);

				stringstream_VIAEH << to_string(lambda) << " " << duration_VIAEH.count() << endl;

				//They should in theory all be epsilon from true value, and therefore, at most be 2 * epsilon from each other
				if (abs_max_diff_vectors(V_approx_solution, V_heap_approx) > (2 * epsilon)){
						printf("DIFFERENCE\n");
				}
				if (abs_max_diff_vectors(V_AE_approx_solution, V_AE_H_approx_solution) > (2 * epsilon)){
						printf("DIFFERENCE\n");
				}
				if (abs_max_diff_vectors(V_bounded_approx_solution, V_approx_solution_upper) > (2 * epsilon)){
						printf("DIFFERENCE\n");
				}

		}

		//WRITE ALL DATA TO THEIR RESPECTVIE FILES	
		write_stringstream_to_file(stringstream_VI, output_stream_VI, file_name_VI);
		write_stringstream_to_file(stringstream_VIU, output_stream_VIU, file_name_VIU);
		write_stringstream_to_file(stringstream_BVI, output_stream_BVI, file_name_BVI);
		write_stringstream_to_file(stringstream_VIH, output_stream_VIH, file_name_VIH);
		write_stringstream_to_file(stringstream_VIAE, output_stream_VIAE, file_name_VIAE);
		write_stringstream_to_file(stringstream_VIAEH, output_stream_VIAEH, file_name_VIAEH);

}

//VIAEH IMPLEMENTATIONS WORK PER ITERATION TEST
void create_data_tables_work_per_iteration_VIAEH_implementations(string filename, int S, int A_num, double epsilon, double gamma, double action_prob, int number_of_transitions, double mean, double variance){

		//FOR WORK PER ITERATION
		//the stringstreams to create the test for the files
		ostringstream stringstream_VIAEH;
		ostringstream stringstream_VIAEH_no_pointers;
		ostringstream stringstream_VIAEH_lazy_update;
		ostringstream stringstream_VIAEH_set;
		ostringstream stringstream_VIAEH_maxmin_heap;
		ostringstream stringstream_VIAEH_approx_lower_bound;

		//the file output objects
		ofstream output_stream_VIAEH;
		ofstream output_stream_VIAEH_no_pointers;
		ofstream output_stream_VIAEH_lazy_update;
		ofstream output_stream_VIAEH_set;
		ofstream output_stream_VIAEH_maxmin_heap;
		ofstream output_stream_VIAEH_approx_lower_bound;
		
		//set the name of the file to write to
		string file_name_VIAEH = "data_tables/work_per_iteration_VIAEH_implementations/" + filename + "_VIAEH.dat";
		string file_name_VIAEH_no_pointers = "data_tables/work_per_iteration_VIAEH_implementations/" + filename + "_VIAEH_no_pointers.dat";
		string file_name_VIAEH_lazy_update = "data_tables/work_per_iteration_VIAEH_implementations/" + filename + "_VIAEH_lazy_update.dat";
		string file_name_VIAEH_set = "data_tables/work_per_iteration_VIAEH_implementations/" + filename + "_VIAEH_set.dat";
		string file_name_VIAEH_maxmin_heap = "data_tables/work_per_iteration_VIAEH_implementations/" + filename + "_VIAEH_maxmin_heap.dat";
		string file_name_VIAEH_approx_lower_bound = "data_tables/work_per_iteration_VIAEH_implementations/" + filename + "_VIAEH_approx_lower_bound.dat";

		//write meta data to all stringstreams as first in their respective files
		//write_meta_data_to_dat_file_work_per_iteration(stringstream_VIAEH, S, A_num, epsilon, gamma, non_zero_transition, upper_reward, action_prob);
		
		//FOR ACCUMULATED WORK
		//the stringstreams to create the test for the files
		ostringstream stringstream_accum_VIAEH;
		ostringstream stringstream_accum_VIAEH_no_pointers;
		ostringstream stringstream_accum_VIAEH_lazy_update;
		ostringstream stringstream_accum_VIAEH_set;
		ostringstream stringstream_accum_VIAEH_maxmin_heap;
		ostringstream stringstream_accum_VIAEH_approx_lower_bound;

		//the file output objects
		ofstream output_stream_accum_VIAEH;
		ofstream output_stream_accum_VIAEH_no_pointers;
		ofstream output_stream_accum_VIAEH_lazy_update;
		ofstream output_stream_accum_VIAEH_set;
		ofstream output_stream_accum_VIAEH_maxmin_heap;
		ofstream output_stream_accum_VIAEH_approx_lower_bound;
		
		//set the name of the file to write to
		string file_name_accum_VIAEH = "data_tables/work_per_iteration_VIAEH_implementations/" + filename + "_accum_VIAEH.dat";
		string file_name_accum_VIAEH_no_pointers = "data_tables/work_per_iteration_VIAEH_implementations/" + filename + "_accum_VIAEH_no_pointers.dat";
		string file_name_accum_VIAEH_lazy_update = "data_tables/work_per_iteration_VIAEH_implementations/" + filename + "_accum_VIAEH_lazy_update.dat";
		string file_name_accum_VIAEH_set = "data_tables/work_per_iteration_VIAEH_implementations/" + filename + "_accum_VIAEH_set.dat";
		string file_name_accum_VIAEH_maxmin_heap = "data_tables/work_per_iteration_VIAEH_implementations/" + filename + "_accum_VIAEH_maxmin_heap.dat";
		string file_name_accum_VIAEH_approx_lower_bound = "data_tables/work_per_iteration_VIAEH_implementations/" + filename + "_accum_VIAEH_approx_lower_bound.dat";

		//FOR ACTION ELIMINATION 
		//the stringstreams to create the test for the files
		ostringstream stringstream_action_elimination_VIAEH;
		ostringstream stringstream_action_elimination_VIAE;
		ostringstream stringstream_action_elimination_VIAEH_no_pointers;
		ostringstream stringstream_action_elimination_VIAEH_lazy_update;
		ostringstream stringstream_action_elimination_VIAEH_set;
		ostringstream stringstream_action_elimination_VIAEH_maxmin_heap;
		ostringstream stringstream_action_elimination_VIAEH_approx_lower_bound;

		//the file output objects
		ofstream output_stream_action_elimination_VIAEH;
		ofstream output_stream_action_elimination_VIAE;
		ofstream output_stream_action_elimination_VIAEH_no_pointers;
		ofstream output_stream_action_elimination_VIAEH_lazy_update;
		ofstream output_stream_action_elimination_VIAEH_set;
		ofstream output_stream_action_elimination_VIAEH_maxmin_heap;
		ofstream output_stream_action_elimination_VIAEH_approx_lower_bound;
		
		//set the name of the file to write to
		string file_name_action_elimination_VIAEH = "data_tables/work_per_iteration_VIAEH_implementations/" + filename + "_action_elimination_VIAEH.dat";
		string file_name_action_elimination_VIAE = "data_tables/work_per_iteration_VIAEH_implementations/" + filename + "_action_elimination_VIAE.dat";
		string file_name_action_elimination_VIAEH_no_pointers = "data_tables/work_per_iteration_VIAEH_implementations/" + filename + "_action_elimination_VIAEH_no_pointers.dat";
		string file_name_action_elimination_VIAEH_lazy_update = "data_tables/work_per_iteration_VIAEH_implementations/" + filename + "_action_elimination_VIAEH_lazy_update.dat";
		string file_name_action_elimination_VIAEH_set = "data_tables/work_per_iteration_VIAEH_implementations/" + filename + "_action_elimination_VIAEH_set.dat";
		string file_name_action_elimination_VIAEH_maxmin_heap = "data_tables/work_per_iteration_VIAEH_implementations/" + filename + "_action_elimination_VIAEH_maxmin_heap.dat";
		string file_name_action_elimination_VIAEH_approx_lower_bound = "data_tables/work_per_iteration_VIAEH_implementations/" + filename + "_action_elimination_VIAEH_approx_lower_bound.dat";

		//write meta data to all stringstreams as first in their respective files
		//write_meta_data_to_dat_file_work_per_iteration(stringstream_accum_VIAEH, S, A_num, epsilon, gamma, non_zero_transition, upper_reward, action_prob);

		//BEGIN EXPERIMENTATION
		//GENERATE THE MDP FROM CURRENT TIME SEED
		int seed = time(0);
		printf("seed: %d\n", seed);

		//TODO permament change to normal distribution here?
		auto MDP = generate_random_MDP_normal_distributed_rewards(S, A_num, action_prob, number_of_transitions, seed, mean, variance);
		R_type R = get<0>(MDP);
		A_type A = get<1>(MDP);
		P_type P = get<2>(MDP);

		//VIAEH
		printf("VIAEH\n");
		A_type A1 = copy_A(A);

		V_type V_AE_H_approx_solution_tuple = value_iteration_action_elimination_heaps(S, R, A1, P, gamma, epsilon);
		vector<double> V_AE_H_approx_solution = get<0>(V_AE_H_approx_solution_tuple);
		int V_AE_H_approx_solution_iterations = get<1>(V_AE_H_approx_solution_tuple);
		vector<microseconds> V_AE_H_approx_solution_work_per_iteration = get<2>(V_AE_H_approx_solution_tuple);
		vector<vector<pair<int,int>>> V_AE_H_approx_solution_actions_eliminated_per_iteration = get<3>(V_AE_H_approx_solution_tuple);

		auto tick_accumulator_VIAEH = V_AE_H_approx_solution_work_per_iteration[0].count();

		for(int iteration = 1; iteration <= V_AE_H_approx_solution_iterations; iteration++) {
				auto iteration_work = V_AE_H_approx_solution_work_per_iteration[iteration].count();
				auto actions_eliminated_iteration = V_AE_H_approx_solution_actions_eliminated_per_iteration[iteration].size();
				tick_accumulator_VIAEH = tick_accumulator_VIAEH + iteration_work;

				stringstream_VIAEH << to_string(iteration) << " " << iteration_work << endl;
				stringstream_action_elimination_VIAEH << to_string(iteration) << " " << actions_eliminated_iteration << endl;
				stringstream_accum_VIAEH << to_string(iteration) << " " << tick_accumulator_VIAEH << endl;
		}

		//VIAE
		printf("VIAE\n");
		A_type A5 = copy_A(A);

		V_type V_AE_approx_solution_tuple = value_iteration_action_elimination(S, R, A5, P, gamma, epsilon);
		vector<double> V_AE_approx_solution = get<0>(V_AE_approx_solution_tuple);
		int V_AE_approx_solution_iterations = get<1>(V_AE_approx_solution_tuple);
		vector<microseconds> V_AE_approx_solution_work_per_iteration = get<2>(V_AE_approx_solution_tuple);
		vector<vector<pair<int,int>>> V_AE_approx_solution_actions_eliminated_per_iteration = get<3>(V_AE_approx_solution_tuple);

		auto tick_accumulator_VIAE = V_AE_approx_solution_work_per_iteration[0].count();

		for(int iteration = 1; iteration <= V_AE_approx_solution_iterations; iteration++) {
				auto iteration_work = V_AE_approx_solution_work_per_iteration[iteration].count();
				auto actions_eliminated_iteration = V_AE_approx_solution_actions_eliminated_per_iteration[iteration].size();
				tick_accumulator_VIAE = tick_accumulator_VIAE + iteration_work;

				//stringstream_VIAE << to_string(iteration) << " " << iteration_work << endl;
				stringstream_action_elimination_VIAE << to_string(iteration) << " " << actions_eliminated_iteration << endl;
				//stringstream_accum_VIAE << to_string(iteration) << " " << tick_accumulator_VIAE << endl;
		}

		//VIAEH_no_pointers
		printf("VIAEH_no_pointers\n");
		A_type A2 = copy_A(A);

		V_type V_AE_H_no_pointers_approx_solution_tuple = value_iteration_action_elimination_heaps_no_pointers(S, R, A2, P, gamma, epsilon);
		vector<double> V_AE_H_no_pointers_approx_solution = get<0>(V_AE_H_no_pointers_approx_solution_tuple);
		int V_AE_H_no_pointers_approx_solution_iterations = get<1>(V_AE_H_no_pointers_approx_solution_tuple);
		vector<microseconds> V_AE_H_no_pointers_approx_solution_work_per_iteration = get<2>(V_AE_H_no_pointers_approx_solution_tuple);
		vector<vector<pair<int,int>>> V_AE_H_no_pointers_approx_solution_actions_eliminated_per_iteration = get<3>(V_AE_H_no_pointers_approx_solution_tuple);

		auto tick_accumulator_VIAEH_no_pointers = V_AE_H_no_pointers_approx_solution_work_per_iteration[0].count();

		for(int iteration = 1; iteration <= V_AE_H_no_pointers_approx_solution_iterations; iteration++) {
				auto iteration_work = V_AE_H_no_pointers_approx_solution_work_per_iteration[iteration].count();
				auto actions_eliminated_iteration = V_AE_H_no_pointers_approx_solution_actions_eliminated_per_iteration[iteration].size();
				tick_accumulator_VIAEH_no_pointers = tick_accumulator_VIAEH_no_pointers + iteration_work;

				stringstream_VIAEH_no_pointers << to_string(iteration) << " " << iteration_work << endl;
				stringstream_action_elimination_VIAEH_no_pointers << to_string(iteration) << " " << actions_eliminated_iteration << endl;
				stringstream_accum_VIAEH_no_pointers << to_string(iteration) << " " << tick_accumulator_VIAEH_no_pointers << endl;
		}
		
		//VIAEH_lazy_update
		printf("VIAEH_lazy_update\n");
		A_type A3 = copy_A(A);
		

		V_type V_AE_H_lazy_update_approx_solution_tuple = value_iteration_action_elimination_heaps_lazy_update(S, R, A3, P, gamma, epsilon);
		vector<double> V_AE_H_lazy_update_approx_solution = get<0>(V_AE_H_lazy_update_approx_solution_tuple);
		int V_AE_H_lazy_update_approx_solution_iterations = get<1>(V_AE_H_lazy_update_approx_solution_tuple);
		vector<microseconds> V_AE_H_lazy_update_approx_solution_work_per_iteration = get<2>(V_AE_H_lazy_update_approx_solution_tuple);
		vector<vector<pair<int,int>>> V_AE_H_lazy_update_approx_solution_actions_eliminated_per_iteration = get<3>(V_AE_H_lazy_update_approx_solution_tuple);

		auto tick_accumulator_VIAEH_lazy_update = V_AE_H_lazy_update_approx_solution_work_per_iteration[0].count();

		for(int iteration = 1; iteration <= V_AE_H_lazy_update_approx_solution_iterations; iteration++) {
				auto iteration_work = V_AE_H_lazy_update_approx_solution_work_per_iteration[iteration].count();
				auto actions_eliminated_iteration = V_AE_H_lazy_update_approx_solution_actions_eliminated_per_iteration[iteration].size();
				tick_accumulator_VIAEH_lazy_update = tick_accumulator_VIAEH_lazy_update + iteration_work;

				stringstream_VIAEH_lazy_update << to_string(iteration) << " " << iteration_work << endl;
				stringstream_action_elimination_VIAEH_lazy_update << to_string(iteration) << " " << actions_eliminated_iteration << endl;
				stringstream_accum_VIAEH_lazy_update << to_string(iteration) << " " << tick_accumulator_VIAEH_lazy_update << endl;
		}
		
		//VIAEH_set
		printf("VIAEH_set\n");
		A_type A4 = copy_A(A);

		V_type V_AE_H_set_approx_solution_tuple = value_iteration_action_elimination_heaps_set(S, R, A4, P, gamma, epsilon);
		vector<double> V_AE_H_set_approx_solution = get<0>(V_AE_H_set_approx_solution_tuple);
		int V_AE_H_set_approx_solution_iterations = get<1>(V_AE_H_set_approx_solution_tuple);
		vector<microseconds> V_AE_H_set_approx_solution_work_per_iteration = get<2>(V_AE_H_set_approx_solution_tuple);
		vector<vector<pair<int,int>>> V_AE_H_set_approx_solution_actions_eliminated_per_iteration = get<3>(V_AE_H_set_approx_solution_tuple);

		auto tick_accumulator_VIAEH_set = V_AE_H_set_approx_solution_work_per_iteration[0].count();

		for(int iteration = 1; iteration <= V_AE_H_set_approx_solution_iterations; iteration++) {
				auto iteration_work = V_AE_H_set_approx_solution_work_per_iteration[iteration].count();
				auto actions_eliminated_iteration = V_AE_H_set_approx_solution_actions_eliminated_per_iteration[iteration].size();
				tick_accumulator_VIAEH_set = tick_accumulator_VIAEH_set + iteration_work;

				stringstream_VIAEH_set << to_string(iteration) << " " << iteration_work << endl;
				stringstream_action_elimination_VIAEH_set << to_string(iteration) << " " << actions_eliminated_iteration << endl;
				stringstream_accum_VIAEH_set << to_string(iteration) << " " << tick_accumulator_VIAEH_set << endl;
		}
		
		//VIAEH_maxmin_heap
		printf("VIAEH_maxmin_heap\n");
		A_type A6 = copy_A(A);

		V_type V_AE_H_maxmin_heap_approx_solution_tuple = value_iteration_action_elimination_heaps_max_min_heap(S, R, A6, P, gamma, epsilon);
		vector<double> V_AE_H_maxmin_heap_approx_solution = get<0>(V_AE_H_maxmin_heap_approx_solution_tuple);
		int V_AE_H_maxmin_heap_approx_solution_iterations = get<1>(V_AE_H_maxmin_heap_approx_solution_tuple);
		vector<microseconds> V_AE_H_maxmin_heap_approx_solution_work_per_iteration = get<2>(V_AE_H_maxmin_heap_approx_solution_tuple);
		vector<vector<pair<int,int>>> V_AE_H_maxmin_heap_approx_solution_actions_eliminated_per_iteration = get<3>(V_AE_H_maxmin_heap_approx_solution_tuple);

		auto tick_accumulator_VIAEH_maxmin_heap = V_AE_H_maxmin_heap_approx_solution_work_per_iteration[0].count();

		for(int iteration = 1; iteration <= V_AE_H_maxmin_heap_approx_solution_iterations; iteration++) {
				auto iteration_work = V_AE_H_maxmin_heap_approx_solution_work_per_iteration[iteration].count();
				auto actions_eliminated_iteration = V_AE_H_maxmin_heap_approx_solution_actions_eliminated_per_iteration[iteration].size();
				tick_accumulator_VIAEH_maxmin_heap = tick_accumulator_VIAEH_maxmin_heap + iteration_work;

				stringstream_VIAEH_maxmin_heap << to_string(iteration) << " " << iteration_work << endl;
				stringstream_action_elimination_VIAEH_maxmin_heap << to_string(iteration) << " " << actions_eliminated_iteration << endl;
				stringstream_accum_VIAEH_maxmin_heap << to_string(iteration) << " " << tick_accumulator_VIAEH_maxmin_heap << endl;
		}

		//VIAEH_approx_lower_bound
		printf("VIAEH_approx_lower_bound\n");
		A_type A7 = copy_A(A);

		V_type V_AE_H_approx_lower_bound_approx_solution_tuple = value_iteration_action_elimination_heaps_lower_bound_approx(S, R, A7, P, gamma, epsilon);
		vector<double> V_AE_H_approx_lower_bound_approx_solution = get<0>(V_AE_H_approx_lower_bound_approx_solution_tuple);
		int V_AE_H_approx_lower_bound_approx_solution_iterations = get<1>(V_AE_H_approx_lower_bound_approx_solution_tuple);
		vector<microseconds> V_AE_H_approx_lower_bound_approx_solution_work_per_iteration = get<2>(V_AE_H_approx_lower_bound_approx_solution_tuple);
		vector<vector<pair<int,int>>> V_AE_H_approx_lower_bound_approx_solution_actions_eliminated_per_iteration = get<3>(V_AE_H_approx_lower_bound_approx_solution_tuple);

		auto tick_accumulator_VIAEH_approx_lower_bound = V_AE_H_approx_lower_bound_approx_solution_work_per_iteration[0].count();

		for(int iteration = 1; iteration <= V_AE_H_approx_lower_bound_approx_solution_iterations; iteration++) {
				auto iteration_work = V_AE_H_approx_lower_bound_approx_solution_work_per_iteration[iteration].count();
				auto actions_eliminated_iteration = V_AE_H_approx_lower_bound_approx_solution_actions_eliminated_per_iteration[iteration].size();
				tick_accumulator_VIAEH_approx_lower_bound = tick_accumulator_VIAEH_approx_lower_bound + iteration_work;

				stringstream_VIAEH_approx_lower_bound << to_string(iteration) << " " << iteration_work << endl;
				stringstream_action_elimination_VIAEH_approx_lower_bound << to_string(iteration) << " " << actions_eliminated_iteration << endl;
				stringstream_accum_VIAEH_approx_lower_bound << to_string(iteration) << " " << tick_accumulator_VIAEH_approx_lower_bound << endl;
		}

		printf("Difference: %f\n", abs_max_diff_vectors(V_AE_H_lazy_update_approx_solution, V_AE_H_no_pointers_approx_solution));
		printf("Difference: %f\n", abs_max_diff_vectors(V_AE_H_lazy_update_approx_solution, V_AE_H_approx_solution));
		printf("Difference: %f\n", abs_max_diff_vectors(V_AE_H_set_approx_solution, V_AE_H_approx_solution));
		printf("Difference: %f\n", abs_max_diff_vectors(V_AE_approx_solution, V_AE_H_no_pointers_approx_solution));
		printf("Difference: %f\n", abs_max_diff_vectors(V_AE_H_maxmin_heap_approx_solution, V_AE_H_no_pointers_approx_solution));
		printf("Difference: %f\n", abs_max_diff_vectors(V_AE_H_approx_lower_bound_approx_solution, V_AE_H_no_pointers_approx_solution));
		
		//WRITE ALL DATA TO THEIR RESPECTVIE FILES	
		write_stringstream_to_file(stringstream_VIAEH, output_stream_VIAEH, file_name_VIAEH);
		write_stringstream_to_file(stringstream_VIAEH_no_pointers, output_stream_VIAEH_no_pointers, file_name_VIAEH_no_pointers);
		write_stringstream_to_file(stringstream_VIAEH_lazy_update, output_stream_VIAEH_lazy_update, file_name_VIAEH_lazy_update);
		write_stringstream_to_file(stringstream_VIAEH_set, output_stream_VIAEH_set, file_name_VIAEH_set);
		write_stringstream_to_file(stringstream_VIAEH_maxmin_heap, output_stream_VIAEH_maxmin_heap, file_name_VIAEH_maxmin_heap);
		write_stringstream_to_file(stringstream_VIAEH_approx_lower_bound, output_stream_VIAEH_approx_lower_bound, file_name_VIAEH_approx_lower_bound);

		write_stringstream_to_file(stringstream_accum_VIAEH, output_stream_accum_VIAEH, file_name_accum_VIAEH);
		write_stringstream_to_file(stringstream_accum_VIAEH_no_pointers, output_stream_accum_VIAEH_no_pointers, file_name_accum_VIAEH_no_pointers);
		write_stringstream_to_file(stringstream_accum_VIAEH_lazy_update, output_stream_accum_VIAEH_lazy_update, file_name_accum_VIAEH_lazy_update);
		write_stringstream_to_file(stringstream_accum_VIAEH_set, output_stream_accum_VIAEH_set, file_name_accum_VIAEH_set);
		write_stringstream_to_file(stringstream_accum_VIAEH_maxmin_heap, output_stream_accum_VIAEH_maxmin_heap, file_name_accum_VIAEH_maxmin_heap);
		write_stringstream_to_file(stringstream_accum_VIAEH_approx_lower_bound, output_stream_accum_VIAEH_approx_lower_bound, file_name_accum_VIAEH_approx_lower_bound);

		write_stringstream_to_file(stringstream_action_elimination_VIAE, output_stream_action_elimination_VIAE, file_name_action_elimination_VIAE);
		write_stringstream_to_file(stringstream_action_elimination_VIAEH, output_stream_action_elimination_VIAEH, file_name_action_elimination_VIAEH);
		write_stringstream_to_file(stringstream_action_elimination_VIAEH_no_pointers, output_stream_action_elimination_VIAEH_no_pointers, file_name_action_elimination_VIAEH_no_pointers);
		write_stringstream_to_file(stringstream_action_elimination_VIAEH_lazy_update, output_stream_action_elimination_VIAEH_lazy_update, file_name_action_elimination_VIAEH_lazy_update);
		write_stringstream_to_file(stringstream_action_elimination_VIAEH_set, output_stream_action_elimination_VIAEH_set, file_name_action_elimination_VIAEH_set);
		write_stringstream_to_file(stringstream_action_elimination_VIAEH_maxmin_heap, output_stream_action_elimination_VIAEH_maxmin_heap, file_name_action_elimination_VIAEH_maxmin_heap);
		write_stringstream_to_file(stringstream_action_elimination_VIAEH_approx_lower_bound, output_stream_action_elimination_VIAEH_approx_lower_bound, file_name_action_elimination_VIAEH_approx_lower_bound);
}

void create_data_tables_bounds_comparisons(string filename, int S, int A_num, double epsilon, double gamma, double action_prob, int number_of_transitions, double mean, double variance){

		//FOR WORK PER ITERATION
		//the stringstreams to create the test for the files
		ostringstream stringstream_VIAE_improved_bounds;
		ostringstream stringstream_VIAE_old_bounds;

		//the file output objects
		ofstream output_stream_VIAE_improved_bounds;
		ofstream output_stream_VIAE_old_bounds;
		
		//set the name of the file to write to
		string file_name_VIAE_improved_bounds = "data_tables/bounds_comparisons/" + filename + "_VIAE_improved_bounds.dat";
		string file_name_VIAE_old_bounds = "data_tables/bounds_comparisons/" + filename + "_VIAE_old_bounds.dat";

		//write meta data to all stringstreams as first in their respective files
		//write_meta_data_to_dat_file_work_per_iteration(stringstream_VIAE_improved_bounds, S, A_num, epsilon, gamma, non_zero_transition, upper_reward, action_prob);
		
		//FOR ACCUMULATED WORK
		//the stringstreams to create the test for the files
		ostringstream stringstream_accum_VIAE_improved_bounds;
		ostringstream stringstream_accum_VIAE_old_bounds;

		//the file output objects
		ofstream output_stream_accum_VIAE_improved_bounds;
		ofstream output_stream_accum_VIAE_old_bounds;
		
		//set the name of the file to write to
		string file_name_accum_VIAE_improved_bounds = "data_tables/bounds_comparisons/" + filename + "_accum_VIAE_improved_bounds.dat";
		string file_name_accum_VIAE_old_bounds = "data_tables/bounds_comparisons/" + filename + "_accum_VIAE_old_bounds.dat";

		//FOR ACTION ELIMINATION 
		//the stringstreams to create the test for the files
		ostringstream stringstream_action_elimination_VIAE_improved_bounds;
		ostringstream stringstream_action_elimination_VIAE_old_bounds;

		//the file output objects
		ofstream output_stream_action_elimination_VIAE_improved_bounds;
		ofstream output_stream_action_elimination_VIAE_old_bounds;
		
		//set the name of the file to write to
		string file_name_action_elimination_VIAE_improved_bounds = "data_tables/bounds_comparisons/" + filename + "_action_elimination_VIAE_improved_bounds.dat";
		string file_name_action_elimination_VIAE_old_bounds = "data_tables/bounds_comparisons/" + filename + "_action_elimination_VIAE_old_bounds.dat";

		//write meta data to all stringstreams as first in their respective files
		//write_meta_data_to_dat_file_work_per_iteration(stringstream_accum_VIAEH, S, A_num, epsilon, gamma, non_zero_transition, upper_reward, action_prob);

		//BEGIN EXPERIMENTATION
		//GENERATE THE MDP FROM CURRENT TIME SEED
		int seed = time(0);
		printf("seed: %d\n", seed);

		//auto MDP = generate_random_MDP_with_variable_parameters_fixed_nonzero_trans_states(S, A_num, action_prob, number_of_transitions, 100.0, seed);
		//auto MDP = generate_random_MDP_normal_distributed_rewards(S, A_num, action_prob, number_of_transitions, seed, mean, variance);
		auto MDP = generate_random_MDP_exponential_distributed_rewards(S, A_num, action_prob, number_of_transitions, 0.02, seed);
		R_type R = get<0>(MDP);
		A_type A = get<1>(MDP);
		P_type P = get<2>(MDP);

		//VIAE_improved_bounds
		printf("VIAE_improved_bounds\n");
		A_type A1 = copy_A(A);

		V_type VIAE_improved_bounds_solution_tuple = value_iteration_action_elimination_improved_bounds(S, R, A1, P, gamma, epsilon);
		vector<double> VIAE_improved_bounds_solution = get<0>(VIAE_improved_bounds_solution_tuple);
		int VIAE_improved_bounds_solution_iterations = get<1>(VIAE_improved_bounds_solution_tuple);
		vector<microseconds> VIAE_improved_bounds_solution_work_per_iteration = get<2>(VIAE_improved_bounds_solution_tuple);
		vector<vector<pair<int,int>>> VIAE_improved_bounds_solution_actions_eliminated_per_iteration = get<3>(VIAE_improved_bounds_solution_tuple);

		auto tick_accumulator_VIAE_improved_bounds = VIAE_improved_bounds_solution_work_per_iteration[0].count();

		for(int iteration = 1; iteration <= VIAE_improved_bounds_solution_iterations; iteration++) {
				auto iteration_work = VIAE_improved_bounds_solution_work_per_iteration[iteration].count();
				auto actions_eliminated_iteration = VIAE_improved_bounds_solution_actions_eliminated_per_iteration[iteration].size();
				tick_accumulator_VIAE_improved_bounds = tick_accumulator_VIAE_improved_bounds + iteration_work;

				stringstream_VIAE_improved_bounds << to_string(iteration) << " " << iteration_work << endl;
				stringstream_action_elimination_VIAE_improved_bounds << to_string(iteration) << " " << actions_eliminated_iteration << endl;
				stringstream_accum_VIAE_improved_bounds << to_string(iteration) << " " << tick_accumulator_VIAE_improved_bounds << endl;
		}

		//VIAE_old_bounds
		printf("VIAE_old_bounds\n");
		A_type A2 = copy_A(A);

		V_type VIAE_old_bounds_solution_tuple = value_iteration_action_elimination_old_bounds(S, R, A2, P, gamma, epsilon);
		vector<double> VIAE_old_bounds_solution = get<0>(VIAE_old_bounds_solution_tuple);
		int VIAE_old_bounds_solution_iterations = get<1>(VIAE_old_bounds_solution_tuple);
		vector<microseconds> VIAE_old_bounds_solution_work_per_iteration = get<2>(VIAE_old_bounds_solution_tuple);
		vector<vector<pair<int,int>>> VIAE_old_bounds_solution_actions_eliminated_per_iteration = get<3>(VIAE_old_bounds_solution_tuple);

		auto tick_accumulator_VIAE_old_bounds = VIAE_old_bounds_solution_work_per_iteration[0].count();

		for(int iteration = 1; iteration <= VIAE_old_bounds_solution_iterations; iteration++) {
				auto iteration_work = VIAE_old_bounds_solution_work_per_iteration[iteration].count();
				auto actions_eliminated_iteration = VIAE_old_bounds_solution_actions_eliminated_per_iteration[iteration].size();
				tick_accumulator_VIAE_old_bounds = tick_accumulator_VIAE_old_bounds + iteration_work;

				stringstream_VIAE_old_bounds << to_string(iteration) << " " << iteration_work << endl;
				stringstream_action_elimination_VIAE_old_bounds << to_string(iteration) << " " << actions_eliminated_iteration << endl;
				stringstream_accum_VIAE_old_bounds << to_string(iteration) << " " << tick_accumulator_VIAE_old_bounds << endl;
		}
		printf("Solutions:\n");
		for(int s = 0; s < 5; s++){
				printf("state %d: %f\n", s, VIAE_improved_bounds_solution[s]);
				printf("state %d: %f\n", s, VIAE_old_bounds_solution[s]);
		}

		printf("Difference: %f\n", abs_max_diff_vectors(VIAE_improved_bounds_solution, VIAE_old_bounds_solution));
		
		//WRITE ALL DATA TO THEIR RESPECTVIE FILES	
		write_stringstream_to_file(stringstream_VIAE_improved_bounds, output_stream_VIAE_improved_bounds, file_name_VIAE_improved_bounds);
		write_stringstream_to_file(stringstream_VIAE_old_bounds, output_stream_VIAE_old_bounds, file_name_VIAE_old_bounds);

		write_stringstream_to_file(stringstream_accum_VIAE_improved_bounds, output_stream_accum_VIAE_improved_bounds, file_name_accum_VIAE_improved_bounds);
		write_stringstream_to_file(stringstream_accum_VIAE_old_bounds, output_stream_accum_VIAE_old_bounds, file_name_accum_VIAE_old_bounds);

		write_stringstream_to_file(stringstream_action_elimination_VIAE_improved_bounds, output_stream_action_elimination_VIAE_improved_bounds, file_name_action_elimination_VIAE_improved_bounds);
		write_stringstream_to_file(stringstream_action_elimination_VIAE_old_bounds, output_stream_action_elimination_VIAE_old_bounds, file_name_action_elimination_VIAE_old_bounds);
}


//ACTIONS TOUCHED VS ELIMINATED
void create_data_tables_actions_touched(string filename, int S, int A_num, double epsilon, double gamma, double action_prob, int number_of_transitions, double mean, double variance){

		//FOR ACTION ELIMINATION 
		//the stringstreams to create the test for the files
		ostringstream stringstream_action_elimination_VIAEH;
		ostringstream stringstream_action_elimination_accum_VIAEH;
		ostringstream stringstream_implicit_action_elimination_VIH_actions_touched;
		ostringstream stringstream_implicit_action_elimination_accum_VIH_actions_touched;

		ostringstream stringstream_action_elimination_VIH_actions_touched;
		ostringstream stringstream_actions_touched_after_elimination;

		//the file output objects
		ofstream output_stream_action_elimination_VIAEH;
		ofstream output_stream_action_elimination_accum_VIAEH;
		ofstream output_stream_implicit_action_elimination_VIH_actions_touched;
		ofstream output_stream_implicit_action_elimination_accum_VIH_actions_touched;

		ofstream output_stream_action_elimination_VIH_actions_touched;
		ofstream output_stream_actions_touched_after_elimination;
		
		//set the name of the file to write to
		string file_name_action_elimination_VIAEH = "data_tables/actions_touched/" + filename + "_action_elimination_VIAEH.dat";
		string file_name_action_elimination_accum_VIAEH = "data_tables/actions_touched/" + filename + "_action_elimination_accum_VIAEH.dat";
		string file_name_implicit_action_elimination_VIH_actions_touched = "data_tables/actions_touched/" + filename + "_implicit_action_elimination_VIH.dat";
		string file_name_implicit_action_elimination_accum_VIH_actions_touched = "data_tables/actions_touched/" + filename + "_implicit_action_elimination_accum_VIH.dat";

		string file_name_action_elimination_VIH_actions_touched = "data_tables/actions_touched/" + filename + "_action_elimination_VIH_actions_touched.dat";
		string file_name_actions_touched_after_elimination = "data_tables/actions_touched/" + filename + "_actions_touched_after_elimination.dat";

		//BEGIN EXPERIMENTATION
		//GENERATE THE MDP FROM CURRENT TIME SEED
		int seed = time(0);
		printf("seed: %d\n", seed);

		//TODO permament change to normal distribution here?
		//auto MDP = generate_random_MDP_normal_distributed_rewards(S, A_num, action_prob, number_of_transitions, seed, mean, variance);
		auto MDP = generate_random_MDP_exponential_distributed_rewards(S, A_num, action_prob, number_of_transitions, 0.02, seed);
		R_type R = get<0>(MDP);
		A_type A = get<1>(MDP);
		P_type P = get<2>(MDP);

		//VIAEH
		printf("VIAEH\n");
		A_type A1 = copy_A(A);

		V_type V_AE_H_approx_solution_tuple = value_iteration_action_elimination_heaps_no_pointers(S, R, A1, P, gamma, epsilon);
		vector<double> V_AE_H_approx_solution = get<0>(V_AE_H_approx_solution_tuple);
		int V_AE_H_approx_solution_iterations = get<1>(V_AE_H_approx_solution_tuple);

		// THIS IS per iteration, we have (state, action) pairs
		vector<vector<pair<int,int>>> V_AE_H_approx_solution_actions_eliminated_per_iteration = get<3>(V_AE_H_approx_solution_tuple);

		//for(int iteration = 1; iteration <= V_AE_H_approx_solution_iterations; iteration++) {
		//		auto iteration_work = V_AE_H_approx_solution_work_per_iteration[iteration].count();
		//		auto actions_eliminated_iteration = V_AE_H_approx_solution_actions_eliminated_per_iteration[iteration].size();
		//		tick_accumulator_VIAEH = tick_accumulator_VIAEH + iteration_work;

		//		stringstream_action_elimination_VIAEH << to_string(iteration) << " " << actions_eliminated_iteration << endl;
		//}

		//VIAE
		printf("VIH actions touched\n");
		A_type A5 = copy_A(A);

		V_type VIH_actions_touched_approx_solution_tuple = value_iteration_actions_touched(S, R, A5, P, gamma, epsilon);
		vector<double> VIH_actions_touched_approx_solution = get<0>(VIH_actions_touched_approx_solution_tuple);
		int VIH_actions_touched_approx_solution_iterations = get<1>(VIH_actions_touched_approx_solution_tuple);

		// THIS IS per iteration, we have (state, action) pairs
		vector<vector<pair<int,int>>> VIH_actions_touched_approx_solution_actions_eliminated_per_iteration = get<3>(VIH_actions_touched_approx_solution_tuple);

		//RECORD ACTIONS TOUCHED IN EACH ITERATION
		for(int iteration = 1; iteration <= VIH_actions_touched_approx_solution_iterations; iteration++) {
				
				//this is number of actions touched in this iteration - to compare with number of saved per iteration per state, just to set it into perspective
				auto actions_touched_iteration = VIH_actions_touched_approx_solution_actions_eliminated_per_iteration[iteration].size();

				stringstream_action_elimination_VIH_actions_touched << to_string(iteration) << " " << (float(actions_touched_iteration) / float(S)) << endl;
		}

		int max_iteration = max(VIH_actions_touched_approx_solution_iterations, V_AE_H_approx_solution_iterations);
		int min_iteration = min(VIH_actions_touched_approx_solution_iterations, V_AE_H_approx_solution_iterations);


		//WE NOW WANT A PER STATE VIEW OF EACH
		vector<vector<pair<int,int>>> V_AE_H_approx_solution_actions_eliminated_per_state(S);
		for(int iteration = 1; iteration <= V_AE_H_approx_solution_iterations; iteration++) {

				//We iterate through the (s,a) list in each iteration 
				for (auto pair : V_AE_H_approx_solution_actions_eliminated_per_iteration[iteration]){

						//We add (iterations, action)
						V_AE_H_approx_solution_actions_eliminated_per_state[pair.first].push_back(make_pair(iteration, pair.second));
				}
		}

		//WE NOW WANT A PER STATE VIEW OF EACH ACTION TOUCED
		vector<vector<pair<int,int>>> VIH_actions_touched_approx_solution_actions_eliminated_per_state(S);
		for(int iteration = 1; iteration <= VIH_actions_touched_approx_solution_iterations; iteration++) {

				//We iterate through the (s,a) list in each iteration 
				for (auto pair : VIH_actions_touched_approx_solution_actions_eliminated_per_iteration[iteration]){

						//We add (iterations, action)
						VIH_actions_touched_approx_solution_actions_eliminated_per_state[pair.first].push_back(make_pair(iteration, pair.second));
				}
		}

		//WE NOW LOOK AT EACH STATE AND MATCH ELIMINATION ITERATION WITH ITERATIONS TOUCHED AFTER THAT
		//consists of (state, action) pairs
		vector<vector<pair<int,int>>> action_touched_after_elimination_per_iteration(max_iteration + 1);
		for(int s = 0; s < S; s++){
				vector<pair<int, int>> VIAEH_per_state = V_AE_H_approx_solution_actions_eliminated_per_state[s];
				vector<pair<int, int>> VIH_per_state = VIH_actions_touched_approx_solution_actions_eliminated_per_state[s];

				for (auto pair_VIAEH : VIAEH_per_state){
						for (auto pair_VIH : VIH_per_state){
								
								//check if action is the same AND elimination iteration is before touch iteration
								if((pair_VIAEH.second == pair_VIH.second) && (pair_VIAEH.first < pair_VIH.first)) {
										action_touched_after_elimination_per_iteration[pair_VIH.first].push_back(pair_VIH);
								}
						}	
				}
		}

		for(int iteration = 1; iteration <= max_iteration; iteration++) {
				auto number_to_write_to_file = action_touched_after_elimination_per_iteration[iteration].size();
				stringstream_actions_touched_after_elimination << to_string(iteration) << " " << (float(number_to_write_to_file) / float(S)) << endl;
		}

		//PREPARE FOR IMPLICIT- VS EXPLICIT ACTION ELIMINATION NUMBER OF ACTION "ELIMINTED" (last iteration where it is never again considered) PER ITERATION
		//AND ACCUMULATION PLOT

		//recording actions eliminated through VIAEH
		int actions_eliminated_accumulator_VIAEH = 0;
		for(int iteration = 1; iteration <= V_AE_H_approx_solution_iterations; iteration++) {
				auto actions_eliminated_iteration = int(V_AE_H_approx_solution_actions_eliminated_per_iteration[iteration].size());
				actions_eliminated_accumulator_VIAEH = actions_eliminated_accumulator_VIAEH + actions_eliminated_iteration;

				stringstream_action_elimination_VIAEH << to_string(iteration) << " " << actions_eliminated_iteration << endl;
				stringstream_action_elimination_accum_VIAEH << to_string(iteration) << " " << actions_eliminated_accumulator_VIAEH << endl;
		}
		
		//recording actions eliminated "implicitly" through VIH
		
		vector<vector<pair<int,int>>> VIH_actions_touched_actions_implicitly_eliminated_per_iteration(VIH_actions_touched_approx_solution_iterations + 1);

		//record if action a in state s is touched in last iteration! (S, A(s)) matrix
		//-1 = not seen yet, so record iteartion number when seen
		//-2 = in last iteration, dont consider when going backwards in iterations
		vector<vector<int>> touched_in_last_iteration;
		int A_max = find_max_A(A) + 1;
		for (int s = 0; s < S; s++){
				//record not yet seen
				vector<int> for_As_iteration(A_max, -1);
				touched_in_last_iteration.push_back(for_As_iteration);
		}

		for (auto pair : action_touched_after_elimination_per_iteration[VIH_actions_touched_approx_solution_iterations]){
				//record that this (s,a) is in the last iteration, such that it is NEVER eliminated	
				touched_in_last_iteration[pair.first][pair.second] = -2;	
		}

		for (int iteration = VIH_actions_touched_approx_solution_iterations - 1; iteration >= 1; iteration-- ){
				for (auto pair : VIH_actions_touched_approx_solution_actions_eliminated_per_iteration[iteration]){
						if (touched_in_last_iteration[pair.first][pair.second] == -1) {
								touched_in_last_iteration[pair.first][pair.second] = iteration;
								VIH_actions_touched_actions_implicitly_eliminated_per_iteration[iteration].push_back(pair);
						}
				}
		}

		int actions_eliminated_accumulator_VIH_actions_touched = 0;
		for(int iteration = 1; iteration <= VIH_actions_touched_approx_solution_iterations; iteration++) {
				auto actions_eliminated_iteration = int(VIH_actions_touched_actions_implicitly_eliminated_per_iteration[iteration].size());
				actions_eliminated_accumulator_VIH_actions_touched = actions_eliminated_accumulator_VIH_actions_touched + actions_eliminated_iteration;

				stringstream_implicit_action_elimination_VIH_actions_touched << to_string(iteration) << " " << actions_eliminated_iteration << endl;
				stringstream_implicit_action_elimination_accum_VIH_actions_touched << to_string(iteration) << " " << actions_eliminated_accumulator_VIH_actions_touched << endl;
		}

		printf("Difference: %f\n", abs_max_diff_vectors(V_AE_H_approx_solution, VIH_actions_touched_approx_solution));
		
		//WRITE ALL DATA TO THEIR RESPECTVIE FILES	
		write_stringstream_to_file(stringstream_action_elimination_VIAEH, output_stream_action_elimination_VIAEH, file_name_action_elimination_VIAEH);
		write_stringstream_to_file(stringstream_action_elimination_accum_VIAEH, output_stream_action_elimination_accum_VIAEH, file_name_action_elimination_accum_VIAEH);
		write_stringstream_to_file(stringstream_implicit_action_elimination_VIH_actions_touched, output_stream_implicit_action_elimination_VIH_actions_touched, file_name_implicit_action_elimination_VIH_actions_touched);
		write_stringstream_to_file(stringstream_implicit_action_elimination_accum_VIH_actions_touched, output_stream_implicit_action_elimination_accum_VIH_actions_touched, file_name_implicit_action_elimination_accum_VIH_actions_touched);

		write_stringstream_to_file(stringstream_action_elimination_VIH_actions_touched, output_stream_action_elimination_VIH_actions_touched, file_name_action_elimination_VIH_actions_touched);
		write_stringstream_to_file(stringstream_actions_touched_after_elimination, output_stream_actions_touched_after_elimination, file_name_actions_touched_after_elimination);
}


//WORK PER ITERATION BEST ALGORITHMS ()
void create_data_tables_work_per_iteration_BEST_implementations(string filename, int S, int A_num, double epsilon, double gamma, double action_prob, int non_zero_transition, double mean, double variance){

		//FOR WORK PER ITERATION
		//the stringstreams to create the test for the files
		ostringstream stringstream_BAO;
		ostringstream stringstream_VIH;
		ostringstream stringstream_VIAEHL;

		//the file output objects
		ofstream output_stream_BAO;
		ofstream output_stream_VIH;
		ofstream output_stream_VIAEHL;
		
		//set the name of the file to write to
		string file_name_BAO = "data_tables/work_per_iteration_BEST_implementations/" + filename + "_BAO.dat"; 
		string file_name_VIH = "data_tables/work_per_iteration_BEST_implementations/" + filename + "_VIH.dat";
		string file_name_VIAEHL = "data_tables/work_per_iteration_BEST_implementations/" + filename + "_VIAEHL.dat";

		//write meta data to all stringstreams as first in their respective files
		//write_meta_data_to_dat_file_work_per_iteration_BEST_implementations(stringstream_BAO, S, A_num, epsilon, gamma, non_zero_transition, action_prob);
		//write_meta_data_to_dat_file_work_per_iteration_BEST_implementations(stringstream_VIH, S, A_num, epsilon, gamma, non_zero_transition, action_prob);
		//write_meta_data_to_dat_file_work_per_iteration_BEST_implementations(stringstream_VIAEHL, S, A_num, epsilon, gamma, non_zero_transition, action_prob);
		
		//FOR ACCUMULATED WORK
		//the stringstreams to create the test for the files
		ostringstream stringstream_accum_BAO;
		ostringstream stringstream_accum_VIH;
		ostringstream stringstream_accum_VIAEHL;

		//the file output objects
		ofstream output_stream_accum_BAO;
		ofstream output_stream_accum_VIH;
		ofstream output_stream_accum_VIAEHL;
		
		//set the name of the file to write to
		string file_name_accum_BAO = "data_tables/work_per_iteration_BEST_implementations/" + filename + "_accum_BAO.dat"; 
		string file_name_accum_VIH = "data_tables/work_per_iteration_BEST_implementations/" + filename + "_accum_VIH.dat";
		string file_name_accum_VIAEHL = "data_tables/work_per_iteration_BEST_implementations/" + filename + "_accum_VIAEHL.dat";

		//write meta data to all stringstreams as first in their respective files
		//write_meta_data_to_dat_file_work_per_iteration(stringstream_accum_BAO, S, A_num, epsilon, gamma, non_zero_transition, action_prob);
		//write_meta_data_to_dat_file_work_per_iteration(stringstream_accum_VIH, S, A_num, epsilon, gamma, non_zero_transition, action_prob);
		//write_meta_data_to_dat_file_work_per_iteration(stringstream_accum_VIAEHL, S, A_num, epsilon, gamma, non_zero_transition, action_prob);

		//BEGIN EXPERIMENTATION
		//GENERATE THE MDP FROM CURRENT TIME SEED
		int seed = time(0);
		printf("seed: %d\n", seed);

		//TODO permament change to normal distribution here?
		//auto MDP = generate_random_MDP_normal_distributed_rewards(S, A_num, action_prob, non_zero_transition, seed, mean, variance);
		auto MDP = generate_random_MDP_exponential_distributed_rewards(S, A_num, action_prob, non_zero_transition, 0.02, seed);
		//auto MDP = generate_random_MDP_with_variable_parameters(S, A_num, action_prob, non_zero_transition, upper_reward, seed);
		R_type R = get<0>(MDP);
		A_type A = get<1>(MDP);
		P_type P = get<2>(MDP);


		//VIH
		printf("VIH\n");
		A_type A2 = copy_A(A);

		V_type V_heap_approx_tuple = value_iteration_with_heap(S, R, A2, P, gamma, epsilon);
		vector<double> V_heap_approx = get<0>(V_heap_approx_tuple);
		int V_heap_approx_iterations = get<1>(V_heap_approx_tuple);
		vector<microseconds> V_heap_approx_work_per_iteration = get<2>(V_heap_approx_tuple);

		auto tick_accumulator_VIH = V_heap_approx_work_per_iteration[0].count();
		
		for(int iteration = 1; iteration <= V_heap_approx_iterations; iteration++) {
				auto iteration_work = V_heap_approx_work_per_iteration[iteration].count();
				tick_accumulator_VIH = tick_accumulator_VIH + iteration_work;

				stringstream_VIH << to_string(iteration) << " " << iteration_work << endl;
				stringstream_accum_VIH << to_string(iteration) << " " << tick_accumulator_VIH << endl;
		}

		//BAO
		printf("BAO\n");
		A_type A3 = copy_A(A);

		V_type BAO_approx_solution_tuple = value_iteration_BAO(S, R, A3, P, gamma, epsilon);
		vector<double> BAO_approx_solution = get<0>(BAO_approx_solution_tuple);
		int BAO_approx_solution_iterations = get<1>(BAO_approx_solution_tuple);
		vector<microseconds> BAO_approx_solution_work_per_iteration = get<2>(BAO_approx_solution_tuple);

		auto tick_accumulator_BAO = BAO_approx_solution_work_per_iteration[0].count();
		
		for(int iteration = 1; iteration <= BAO_approx_solution_iterations; iteration++) {
				auto iteration_work = BAO_approx_solution_work_per_iteration[iteration].count();
				tick_accumulator_BAO = tick_accumulator_BAO + iteration_work;

				stringstream_BAO << to_string(iteration) << " " << iteration_work << endl;
				stringstream_accum_BAO << to_string(iteration) << " " << tick_accumulator_BAO << endl;
		}

		//VIAEHL
		printf("VIAEHL\n");
		A_type A7 = copy_A(A);

		V_type VIAEHL_approx_solution_tuple = value_iteration_action_elimination_heaps_lower_bound_approx(S, R, A7, P, gamma, epsilon);
		vector<double> VIAEHL_approx_solution = get<0>(VIAEHL_approx_solution_tuple);
		int VIAEHL_approx_solution_iterations = get<1>(VIAEHL_approx_solution_tuple);
		vector<microseconds> VIAEHL_approx_solution_work_per_iteration = get<2>(VIAEHL_approx_solution_tuple);

		auto tick_accumulator_VIAEHL = VIAEHL_approx_solution_work_per_iteration[0].count();

		for(int iteration = 1; iteration <= VIAEHL_approx_solution_iterations; iteration++) {
				auto iteration_work = VIAEHL_approx_solution_work_per_iteration[iteration].count();
				tick_accumulator_VIAEHL = tick_accumulator_VIAEHL + iteration_work;

				stringstream_VIAEHL << to_string(iteration) << " " << iteration_work << endl;
				stringstream_accum_VIAEHL << to_string(iteration) << " " << tick_accumulator_VIAEHL << endl;
		}


		//They should in theory all be epsilon from true value, and therefore, at most be 2 * epsilon from each other
		printf("Difference: %f\n", abs_max_diff_vectors(V_heap_approx, BAO_approx_solution));
		printf("Difference: %f\n", abs_max_diff_vectors(V_heap_approx, VIAEHL_approx_solution));
		printf("Difference: %f\n", abs_max_diff_vectors(BAO_approx_solution, VIAEHL_approx_solution));
		

		//WRITE ALL DATA TO THEIR RESPECTVIE FILES	
		write_stringstream_to_file(stringstream_BAO, output_stream_BAO, file_name_BAO);
		write_stringstream_to_file(stringstream_VIH, output_stream_VIH, file_name_VIH);
		write_stringstream_to_file(stringstream_VIAEHL, output_stream_VIAEHL, file_name_VIAEHL);

		write_stringstream_to_file(stringstream_accum_BAO, output_stream_accum_BAO, file_name_accum_BAO);
		write_stringstream_to_file(stringstream_accum_VIH, output_stream_accum_VIH, file_name_accum_VIH);
		write_stringstream_to_file(stringstream_accum_VIAEHL, output_stream_accum_VIAEHL, file_name_accum_VIAEHL);
}

void create_data_tables_number_of_actions_best_implementations(string filename, int S, int A_max, double epsilon, double gamma, double upper_reward, double non_zero_transition){

		//the stringstreams to create the test for the files
		ostringstream stringstream_VIH;
		ostringstream stringstream_VIAEHL;
		ostringstream stringstream_BAO;

		//the file output objects
		ofstream output_stream_VIH;
		ofstream output_stream_VIAEHL;
		ofstream output_stream_BAO;
		
		//set the name of the file to write to
		string file_name_VIH = "data_tables/number_of_actions_best/" + filename + "_VIH.dat";
		string file_name_VIAEHL = "data_tables/number_of_actions_best/" + filename + "_VIAEHL.dat";
		string file_name_BAO = "data_tables/number_of_actions_best/" + filename + "_BAO.dat";

		//The varying parameters
		int A_starting_value = 50;
		int A_finishing_value = A_max;
		int A_increment = 50;

		//hardcoded parameter
		double action_prob = 1.0;

		for (int A_num = A_starting_value; A_num <= A_finishing_value; A_num = A_num + A_increment){
				
				printf("Beginning iteration A_num = %d\n", A_num);

				//auto MDP = generate_random_MDP_normal_distributed_rewards(S, A_num, action_prob, non_zero_transition, seed, 1000, 10);
				//auto MDP = generate_random_MDP_exponential_distributed_rewards(S, A_num, action_prob, non_zero_transition, 0.02, seed);
				//auto MDP = generate_random_MDP_with_variable_parameters(S, A_num, action_prob, non_zero_transition, 1000, seed);
				//GENERATE THE MDP
				int seed = time(0);
				//auto MDP = generate_random_MDP_exponential_distributed_rewards(S, A_num, 1.0, non_zero_transition, 0.02, seed);
				auto MDP = generate_random_MDP_normal_distributed_rewards(S, A_num, 1.0, S, seed, 1000, 10);
				//auto MDP = generate_random_MDP_with_variable_parameters(S, A_num, 1.0, non_zero_transition, upper_reward, seed);
				R_type R = get<0>(MDP);
				A_type A = get<1>(MDP);
				P_type P = get<2>(MDP);


				//VIH testing
				A_type A2 = copy_A(A);
				auto start_VIH = high_resolution_clock::now();

				V_type V_heap_approx_tuple = value_iteration_with_heap(S, R, A2, P, gamma, epsilon);
				vector<double> V_heap_approx = get<0>(V_heap_approx_tuple);

				auto stop_VIH = high_resolution_clock::now();
				auto duration_VIH = duration_cast<microseconds>(stop_VIH - start_VIH);

				stringstream_VIH << to_string(A_num) << " " << duration_VIH.count() << endl;

				//VIAEHL
				A_type A9 = copy_A(A);
				auto start_VIAEHL = high_resolution_clock::now();

				V_type VIAEHL_approx_solution_tuple = value_iteration_action_elimination_heaps_lower_bound_approx(S, R, A9, P, gamma, epsilon);
				vector<double> VIAEHL_approx_solution = get<0>(VIAEHL_approx_solution_tuple);

				auto stop_VIAEHL = high_resolution_clock::now();
				auto duration_VIAEHL = duration_cast<microseconds>(stop_VIAEHL - start_VIAEHL);

				stringstream_VIAEHL << to_string(A_num) << " " << duration_VIAEHL.count() << endl;
				
				//BAO
				A_type A8 = copy_A(A);
				auto start_BAO = high_resolution_clock::now();

				V_type BAO_approx_solution_tuple = value_iteration_BAO(S, R, A8, P, gamma, epsilon);
				vector<double> BAO_approx_solution = get<0>(BAO_approx_solution_tuple);

				auto stop_BAO = high_resolution_clock::now();
				auto duration_BAO = duration_cast<microseconds>(stop_BAO - start_BAO);

				stringstream_BAO << to_string(A_num) << " " << duration_BAO.count() << endl;
			
				//They should in theory all be epsilon from true value, and therefore, at most be 2 * epsilon from each other
				if (abs_max_diff_vectors(VIAEHL_approx_solution	, V_heap_approx) > (2 * epsilon)){
						printf("DIFFERENCE\n");
				}
				if (abs_max_diff_vectors(VIAEHL_approx_solution	, BAO_approx_solution) > (2 * epsilon)){
						printf("DIFFERENCE\n");
				}

		}

		//WRITE ALL DATA TO THEIR RESPECTVIE FILES	
		write_stringstream_to_file(stringstream_VIH, output_stream_VIH, file_name_VIH);
		write_stringstream_to_file(stringstream_VIAEHL, output_stream_VIAEHL, file_name_VIAEHL);
		write_stringstream_to_file(stringstream_BAO, output_stream_BAO, file_name_BAO);
}

void create_data_tables_number_of_states_best_implementations(string filename, int S_max, int A_num, double epsilon, double gamma, double upper_reward, double non_zero_transition){

		//the stringstreams to create the test for the files
		ostringstream stringstream_VIH;
		ostringstream stringstream_VIAEHL;
		ostringstream stringstream_BAO;

		//the file output objects
		ofstream output_stream_VIH;
		ofstream output_stream_VIAEHL;
		ofstream output_stream_BAO;
		
		//set the name of the file to write to
		string file_name_VIH = "data_tables/number_of_states_best/" + filename + "_VIH.dat";
		string file_name_VIAEHL = "data_tables/number_of_states_best/" + filename + "_VIAEHL.dat";
		string file_name_BAO = "data_tables/number_of_states_best/" + filename + "_BAO.dat";

		//The varying parameters
		int S_starting_value = 50;
		int S_finishing_value = S_max;
		int S_increment = 50;

		//hardcoded parameter
		double action_prob = 1.0;

		//write meta data to all stringstreams as first in their respective files
		for (int S = S_starting_value; S <= S_finishing_value; S = S + S_increment){
				
				printf("Beginning iteration S = %d\n", S);

				//GENERATE THE MDP
				int seed = time(0);
				auto MDP = generate_random_MDP_normal_distributed_rewards(S, A_num, action_prob, S, seed, 1000, 10);
				//auto MDP = generate_random_MDP_with_variable_parameters(S, A_num, action_prob, non_zero_transition, upper_reward, seed);
				R_type R = get<0>(MDP);
				A_type A = get<1>(MDP);
				P_type P = get<2>(MDP);
				//VIH testing
				A_type A2 = copy_A(A);
				auto start_VIH = high_resolution_clock::now();

				V_type V_heap_approx_tuple = value_iteration_with_heap(S, R, A2, P, gamma, epsilon);
				vector<double> V_heap_approx = get<0>(V_heap_approx_tuple);

				auto stop_VIH = high_resolution_clock::now();
				auto duration_VIH = duration_cast<microseconds>(stop_VIH - start_VIH);

				stringstream_VIH << to_string(S) << " " << duration_VIH.count() << endl;

				//VIAEHL
				A_type A8 = copy_A(A);
				auto start_VIAEHL = high_resolution_clock::now();

				V_type VIAEHL_approx_solution_tuple = value_iteration_action_elimination_heaps_lower_bound_approx(S, R, A8, P, gamma, epsilon);
				vector<double> VIAEHL_approx_solution = get<0>(VIAEHL_approx_solution_tuple);

				auto stop_VIAEHL = high_resolution_clock::now();
				auto duration_VIAEHL = duration_cast<microseconds>(stop_VIAEHL - start_VIAEHL);

				stringstream_VIAEHL << to_string(S) << " " << duration_VIAEHL.count() << endl;
				
				//BAO
				A_type A9 = copy_A(A);
				auto start_BAO = high_resolution_clock::now();

				V_type BAO_approx_solution_tuple = value_iteration_BAO(S, R, A9, P, gamma, epsilon);
				vector<double> BAO_approx_solution = get<0>(BAO_approx_solution_tuple);

				auto stop_BAO = high_resolution_clock::now();
				auto duration_BAO = duration_cast<microseconds>(stop_BAO - start_BAO);

				stringstream_BAO << to_string(S) << " " << duration_BAO.count() << endl;
				
				//They should in theory all be epsilon from true value, and therefore, at most be 2 * epsilon from each other
				if (abs_max_diff_vectors(VIAEHL_approx_solution, BAO_approx_solution) > (2 * epsilon)){
						printf("DIFFERENCE\n");
				}
				if (abs_max_diff_vectors(VIAEHL_approx_solution, V_heap_approx) > (2 * epsilon)){
						printf("DIFFERENCE\n");
				}

		}

		//WRITE ALL DATA TO THEIR RESPECTVIE FILES	
		write_stringstream_to_file(stringstream_VIH, output_stream_VIH, file_name_VIH);
		write_stringstream_to_file(stringstream_VIAEHL, output_stream_VIAEHL, file_name_VIAEHL);
		write_stringstream_to_file(stringstream_BAO, output_stream_BAO, file_name_BAO);
}

void create_data_tables_number_of_actions_VIH_implementations(string filename, int S, int A_max, double epsilon, double gamma, double upper_reward, double non_zero_transition){

		//the stringstreams to create the test for the files
		ostringstream stringstream_VIH;
		ostringstream stringstream_VIH_custom;

		//the file output objects
		ofstream output_stream_VIH;
		ofstream output_stream_VIH_custom;
		
		//set the name of the file to write to
		string file_name_VIH = "data_tables/number_of_actions_VIH_impl/" + filename + "_VIH.dat";
		string file_name_VIH_custom = "data_tables/number_of_actions_VIH_impl/" + filename + "_VIH_custom.dat";

		//The varying parameters
		int A_starting_value = 50;
		int A_finishing_value = A_max;
		int A_increment = 50;

		//hardcoded parameter
		double action_prob = 1.0;

		for (int A_num = A_starting_value; A_num <= A_finishing_value; A_num = A_num + A_increment){
				
				printf("Beginning iteration A_num = %d\n", A_num);

				//auto MDP = generate_random_MDP_normal_distributed_rewards(S, A_num, action_prob, non_zero_transition, seed, 1000, 10);
				//auto MDP = generate_random_MDP_exponential_distributed_rewards(S, A_num, action_prob, non_zero_transition, 0.02, seed);
				//auto MDP = generate_random_MDP_with_variable_parameters(S, A_num, action_prob, non_zero_transition, 1000, seed);
				//GENERATE THE MDP
				int seed = time(0);
				auto MDP = generate_random_MDP_exponential_distributed_rewards(S, A_num, 1.0, S, 0.02, seed);
				//auto MDP = generate_random_MDP_normal_distributed_rewards(S, A_num, 1.0, S, seed, 1000, 10);
				//auto MDP = generate_random_MDP_with_variable_parameters(S, A_num, 1.0, non_zero_transition, upper_reward, seed);
				R_type R = get<0>(MDP);
				A_type A = get<1>(MDP);
				P_type P = get<2>(MDP);

				//VIH testing
				A_type A2 = copy_A(A);
				auto start_VIH = high_resolution_clock::now();

				V_type V_heap_approx_tuple = value_iteration_with_heap(S, R, A2, P, gamma, epsilon);
				vector<double> V_heap_approx = get<0>(V_heap_approx_tuple);

				auto stop_VIH = high_resolution_clock::now();
				auto duration_VIH = duration_cast<microseconds>(stop_VIH - start_VIH);

				stringstream_VIH << to_string(A_num) << " " << duration_VIH.count() << endl;

				//VIH_custom
				A_type A8 = copy_A(A);
				auto start_VIH_custom = high_resolution_clock::now();

				V_type VIH_custom_approx_solution_tuple = value_iteration_VIH_custom(S, R, A8, P, gamma, epsilon);
				vector<double> VIH_custom_approx_solution = get<0>(VIH_custom_approx_solution_tuple);

				auto stop_VIH_custom = high_resolution_clock::now();
				auto duration_VIH_custom = duration_cast<microseconds>(stop_VIH_custom - start_VIH_custom);

				stringstream_VIH_custom << to_string(A_num) << " " << duration_VIH_custom.count() << endl;
			
				//They should in theory all be epsilon from true value, and therefore, at most be 2 * epsilon from each other
				if (abs_max_diff_vectors(V_heap_approx, VIH_custom_approx_solution) > (2 * epsilon)){
						printf("DIFFERENCE\n");
				}

		}

		//WRITE ALL DATA TO THEIR RESPECTVIE FILES	
		write_stringstream_to_file(stringstream_VIH, output_stream_VIH, file_name_VIH);
		write_stringstream_to_file(stringstream_VIH_custom, output_stream_VIH_custom, file_name_VIH_custom);
}
