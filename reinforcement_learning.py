# Implementation of Reinforcement Learning

# import dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import random


data = pd.read_csv("Section 32 - Upper Confidence Bound (UCB)/Ads_CTR_Optimisation.csv")

# Load the dataset in a DataFrame format

class ReinforcementLearning:

    def __init__(self, data: pd.DataFrame = None, N:int=None, d:int=None):
        """
        Parameters:
            data (DataFrame): The dataset containing the rewards.
            N (int): The number of rounds to run the algorithm.
            d (int): The number of different ads.
        """
        self.data = data
        self.N = N
        self.d = d
        
    # Implementing Upper Confidence Bound   
    def upper_confidence_bound(self):
        """
    Runs the UCB algorithm on the given data.

    Parameters:
        data (DataFrame): The dataset containing the rewards.
        N (int): The number of rounds to run the algorithm.
        d (int): The number of different ads.

    Returns:
        ads_selected (list): A list of ads selected at each round.
        number_of_selections (list): The number of times each ad was selected.
        sum_of_rewards (list): The sum of rewards for each ad.
        total_reward (int): The total reward accumulated.
        upper_bounds (list): The upper bounds at each round for the selected ad.
    """
        number_of_selections = [0] * self.d
        sum_of_rewards = [0] * self.d
        ads_selected = []
        total_reward = 0
        for n in range(0, self.N):
            ad = 0
            max_upper_bound = 0
            for i in range(0, self.d):
                if number_of_selections[i] > 0:
                    average_reward = sum_of_rewards[i] / number_of_selections[i]
                    delta_i = math.sqrt(3/2 * math.log(n+1) / number_of_selections[i])
                    upper_bound = average_reward + delta_i
                else:
                    upper_bound = 1e400
                if upper_bound > max_upper_bound:
                    max_upper_bound = upper_bound
                    ad = i 
            ads_selected.append(ad)
            number_of_selections[ad] += 1
            reward = data.values[n, ad]
            sum_of_rewards[ad] += reward
            total_reward += reward
        
        return number_of_selections, sum_of_rewards, ads_selected, total_reward
    
    # Visualization for Upper Confindence Bound
    def visualization_ucb(self, ads_selected : list):
        plt.hist(ads_selected)
        plt.xticks(range(0, 10))
        plt.title("Histogram of ad selected using UCB")
        plt.ylabel("Number of times each ad was selected")
        plt.xlabel("Ads")
        plt.savefig('ucb_results.png')
        plt.show()
    
    # Implementing Thompson Sampling
    def thompson_sampling(self):
        """
        Parameters
        ----------
        number_of_rewards : list, optional
            DESCRIPTION. The default is None.
        number_of_failures : list, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.
        Step 1:
            At each round n, we consider two numbers of each ad i:
                number_of_rewards : list, optional
                    DESCRIPTION. The default is None.
                number_of_failures : list, optional
                    DESCRIPTION. The default is None.
        Step 2:
            For each ad i we take a random draw from the distribution below
            random.betavariate(number_of_rewards[i] + 1, number_of_failures[i] + 1)
            
        Step 3:
            We take the ad that has the highest theta of i (n)
            
        We also save the total_reward
        """
        
        number_of_rewards = [0] * self.d
        number_of_failures = [0] * self.d
        ads_selected = []
        total_reward = 0
        for n in range(0, self.N):
            ad = 0
            max_random = 0
            for i in range(0, self.d):
                # Random draw thetai(n) from this posterior distribution
                random_beta = random.betavariate(number_of_rewards[i] + 1, number_of_failures[i] + 1)
                if random_beta > max_random:
                    max_random = random_beta
                    ad = i 
            ads_selected.append(ad)
            reward = data.values[n, ad]
            if reward == 1:
                number_of_rewards[ad] += 1
            else:
                number_of_failures[ad] += 1
            total_reward += reward
            
        return number_of_rewards, number_of_failures, ads_selected, total_reward
    
    # Visualization for Thompson Sampling
    def visualization_thompson(self, ads_selected: list):
        # _, _, ads_selected, _ = self.upper_confidence_bound()
        plt.hist(ads_selected)
        plt.xticks(range(0, 10))
        plt.title("Histogram of ad selected using Thompson Sampling")
        plt.ylabel("Number of times of each ad was selected")
        plt.xlabel("Ads")
        plt.savefig('ucb_results.png')
        plt.show()
        
        
