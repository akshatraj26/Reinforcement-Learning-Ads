import pandas as pd

from reinforcement_learning import ReinforcementLearning

df = pd.read_csv("Section 32 - Upper Confidence Bound (UCB)/Ads_CTR_Optimisation.csv")

N = len(df)
d = len(df.columns)

rf = ReinforcementLearning(data=df, N=N, d=d)


number_of_selections, sum_of_rewards, ads_selected, total_reward = rf.upper_confidence_bound()


rf.visualization_ucb(ads_selected)


number_of_rewards, number_of_failures, ads_selected_th, total_reward_th = rf.thompson_sampling()

rf.visualization_thompson(ads_selected_th)


