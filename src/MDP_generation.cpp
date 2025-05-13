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

R_type R_uniform_random_distribution(int S, int A_num, double upper_bound_reward, default_random_engine& e){

		uniform_real_distribution<double> uniform_reward(0, upper_bound_reward);
		R_type R;
		for (int i = 0; i < S; ++i){
				vector<double> R_s;
				for (int j = 0; j < A_num; ++j) {
					double rand_reward = uniform_reward(e);
					R_s.push_back(rand_reward);
				}
				R.push_back(R_s);
		}
		return R;
}

R_type R_normal_distribution(int S, int A_num, double reward_dist_mean, double reward_dist_variance, default_random_engine& e){

		normal_distribution<double> normal_distributed_reward(reward_dist_mean,reward_dist_variance);
		R_type R;
		for (int i = 0; i < S; ++i){
				vector<double> R_s;
				for (int j = 0; j < A_num; ++j) {
					double rand_reward = normal_distributed_reward(e);
					R_s.push_back(rand_reward);
				}
				R.push_back(R_s);
		}
		return R;
}

R_type R_exponential_distribution(int S, int A_num, double lambda, default_random_engine& e){

		exponential_distribution<double> exponential_distributed_reward(lambda);

		R_type R;
		for (int i = 0; i < S; ++i){
				vector<double> R_s;
				for (int j = 0; j < A_num; ++j) {
					double rand_reward = exponential_distributed_reward(e);
					R_s.push_back(rand_reward);
				}
				R.push_back(R_s);
		}
		return R;
}

R_type R_uniform_distribution_reward_prob(int S, int A_num, double upper_bound_reward, double reward_factor, double reward_prob, default_random_engine& e){
		//Generate a reward for each state-action pair
		uniform_real_distribution<double> uniform_reward(0,upper_bound_reward);

		//The prob that an reward is large
		bernoulli_distribution b_reward(reward_prob);

		R_type R;
		for (int i = 0; i < S; ++i){
				vector<double> R_s;
				for (int j = 0; j < A_num; ++j) {
					double rand_reward = uniform_reward(e);

					//Is the reward to be a factor reward_factor larger
					if (b_reward(e)){
							rand_reward = reward_factor * rand_reward;
					}
					R_s.push_back(rand_reward);
				}
				R.push_back(R_s);
		}
		return R;

}

//Generate an action set for each state
A_type A_generate(int S, int A_num, double action_prob, default_random_engine& e){
		
		// bernouli distribution that returns true with probability A_prob
		// TODO: static or not?
		bernoulli_distribution b_A(action_prob);

		//OBS: check if action set is empty and redo if it is. Need to have at least one action.
		A_type A;
		int average_size = 0;
		for (int i = 0; i < S; ++i) {
				vector<int> A_s;
				for (int i = 0; i < A_num; ++i) {
						if (b_A(e)) {
								//The action i is chosen to be in set A(s)
								A_s.push_back(i);	
						}
				}
				//Have insert this to error when there are no actions, which is a mistake!
				if (A_s.size() == 0){
						printf("NO ACTIONS\n");
						A_s.push_back(0);
				}
				average_size += A_s.size();
				A.push_back(A_s);
		}
		double average = double(average_size) / double(S);
		
		return A;
}

P_type P_nonzero_probability(int S, int A_num, double non_zero_prob, default_random_engine& e) {
		//Generate probability distribution P
	
		//To select random 
		uniform_int_distribution<> rand_state(0, S - 1);

		// bernouli distribution that returns true with probability A_prob
		bernoulli_distribution b_P(non_zero_prob);
		//From this random number generator, it generates an double value between 0 and 1, i.e. an probability
		//From this random number generator, it generates an double value between 0 and upper_bound_reward, i.e. an unifrmly random reward
		uniform_real_distribution<double> uniform_prob_dist(0,1);
		P_type P;
		for (int i = 0; i < S; ++i) {
				//we now fix state s
				vector<pair<vector<double>, vector<int> > > P_s;

				//for each action
				for (int i = 0; i < A_num; ++i) {
						//prob dist induced by choosing action a
						vector<double> P_s_a;

						//keeps the states that have a non-zero probability to transition to 
						vector<int> P_s_a_nonzero_states;
						for (int j = 0; j < S; ++j) {
								bool a_is_nonzero = b_P(e);
								if (a_is_nonzero) {
										double trans_prob = uniform_prob_dist(e);
										P_s_a.push_back(trans_prob);

										//add the state index to the vector of nonzero states
										P_s_a_nonzero_states.push_back(j);
								} else {
										P_s_a.push_back(double(0));
								}
						}
						//make it a distribution that sums to 
						double sum = accumulate(P_s_a.begin(), P_s_a.end(), 0.0);
						if (sum == 0.0){
								printf("NO TRANSITIONS\n");
								int rand_trans_state = rand_state(e);
								printf("state %d\n", rand_trans_state);
								P_s_a[rand_trans_state] = 1.0;
						}
						for (int j = 0; j < S; ++j) {
								P_s_a[j] *= (1.0 / sum);
						}
						
						//TODO use emplace_back here instead for better performance
						P_s.push_back(make_pair(P_s_a, P_s_a_nonzero_states));
				}
				P.push_back(P_s);
		}

		return P;

}

P_type P_fixed_size(int S, int A_num, int num_of_nonzero_transition_states, default_random_engine& e) {

		//Generate probability distribution P
	
		//To select random 
		uniform_int_distribution<> rand_state(0, S - 1);

		//From this random number generator, it generates an double value between 0 and 1, i.e. an probability
		//From this random number generator, it generates an double value between 0 and upper_bound_reward, i.e. an unifrmly random reward
		uniform_real_distribution<double> uniform_prob_dist(0,1);
		P_type P;
		for (int i = 0; i < S; ++i) {
				//we now fix state s
				vector<pair<vector<double>, vector<int> > > P_s;

				//for each action
				for (int i = 0; i < A_num; ++i) {
						//prob dist induced by choosing action a
						//init with all zeros and only change those that are nonzero
						vector<double> P_s_a(S, double(0));

						//keeps the states that have a non-zero probability to transition to 
						
						//generate list of indicies and shuffle using the random engine e defined above
						int states[S];
						for (int i = 0; i < S; i++) {
								states[i] = i;
						}
						shuffle(states, states + S, e);
						vector<int> P_s_a_nonzero_states(states, states + num_of_nonzero_transition_states);
						sort(P_s_a_nonzero_states.begin(), P_s_a_nonzero_states.end());
						
						//give the nonzero states probabilities
						for (int nonzero_state : P_s_a_nonzero_states) {
								double trans_prob = uniform_prob_dist(e);
								P_s_a[nonzero_state] = trans_prob;
						}

						//make it a distribution that sums to 
						double sum = accumulate(P_s_a.begin(), P_s_a.end(), 0.0);
						for (int j = 0; j < S; ++j) {
								P_s_a[j] *= (1.0 / sum);
						}
						
						//TODO use emplace_back here instead for better performance
						P_s.push_back(make_pair(P_s_a, P_s_a_nonzero_states));
				}
				P.push_back(P_s);
		}

		return P;
}

MDP_type generate_random_MDP_with_variable_parameters(int S, int A_num, double action_prob, double non_zero_prob, double upper_bound_reward, int seed){

		//make random engine
		static default_random_engine e(seed);

		R_type R = R_uniform_random_distribution(S, A_num, upper_bound_reward, e);
		A_type A = A_generate(S, A_num, action_prob, e);
		P_type P = P_nonzero_probability(S, A_num, non_zero_prob, e);

		MDP_type MDP = make_tuple(R, A, P);

		return MDP;
}

//with some probability, make each reward a factor larger than the others
MDP_type generate_random_MDP_with_variable_parameters_and_reward(int S, int A_num, double action_prob, double non_zero_prob, double reward_factor, double reward_prob, double upper_bound_reward, int seed){

		//We give it a seed to control the generation
		static default_random_engine e(seed);

		R_type R = R_uniform_distribution_reward_prob(S, A_num, upper_bound_reward, reward_factor, reward_prob, e);
		A_type A = A_generate(S, A_num, action_prob, e);
		P_type P = P_nonzero_probability(S, A_num, non_zero_prob, e);

		MDP_type MDP = make_tuple(R, A, P);

		return MDP;
}

//fixed number of non-zero transition states in each state
MDP_type generate_random_MDP_with_variable_parameters_fixed_nonzero_trans_states(int S, int A_num, double action_prob, int num_of_nonzero_transition_states, double upper_bound_reward, int seed){

		//We give it a seed to control the generation
		static default_random_engine e(seed);

		R_type R = R_uniform_random_distribution(S, A_num, upper_bound_reward, e);
		A_type A = A_generate(S, A_num, action_prob, e);
		P_type P = P_fixed_size(S, A_num, num_of_nonzero_transition_states, e);

		MDP_type MDP = make_tuple(R, A, P);

		return MDP;
}

MDP_type generate_random_MDP_normal_distributed_rewards(int S, int A_num, double action_prob, int num_of_nonzero_transition_states, int seed, double reward_dist_mean, double reward_dist_variance){

		static default_random_engine e(seed);

		R_type R = R_normal_distribution(S, A_num, reward_dist_mean, reward_dist_variance, e);
		A_type A = A_generate(S, A_num, action_prob, e);
		P_type P = P_fixed_size(S, A_num, num_of_nonzero_transition_states, e);

		MDP_type MDP = make_tuple(R, A, P);

		return MDP;
}

MDP_type generate_random_MDP_exponential_distributed_rewards(int S, int A_num, double action_prob, int num_of_nonzero_transition_states, double lambda, int seed){

		static default_random_engine e(seed);

		R_type R = R_exponential_distribution(S, A_num, lambda, e);
		A_type A = A_generate(S, A_num, action_prob, e);
		P_type P = P_fixed_size(S, A_num, num_of_nonzero_transition_states, e);

		MDP_type MDP = make_tuple(R, A, P);

		return MDP;
}
