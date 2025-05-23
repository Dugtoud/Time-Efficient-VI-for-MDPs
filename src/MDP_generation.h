#ifndef MDP_GENERATION_H
#define MDP_GENERATION_H

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

MDP_type generate_random_MDP_with_variable_parameters(int S, int A_num, double action_prob, double non_zero_prob, double upper_bound_reward, int seed);

MDP_type generate_random_MDP_with_variable_parameters_and_reward(int S, int A_num, double action_prob, double non_zero_prob, double reward_factor, double reward_prob, double upper_bound_reward, int seed);

MDP_type generate_random_MDP_with_variable_parameters_fixed_nonzero_trans_states(int S, int A_num, double action_prob, int num_of_nonzero_transition_states, double upper_bound_reward, int seed);

MDP_type generate_random_MDP_normal_distributed_rewards(int S, int A_num, double action_prob, int num_of_nonzero_transition_states, int seed, double reward_dist_mean, double reward_dist_variance);

MDP_type generate_random_MDP_exponential_distributed_rewards(int S, int A_num, double action_prob, int num_of_nonzero_transition_states, double lambda, int seed);
#endif
