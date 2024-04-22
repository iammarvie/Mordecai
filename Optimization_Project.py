# Mordecai Ethapemi
# Date: 04/18/2024
# Description: This program is a simple optimization project that uses the scipy.optimize library to find the minimum value of a function.
# The goal of optimizing is to determine the best team i can make using minimal budget. The budget is 100 million dollars and the players have different values.
# The players are divided into 4 categories: Goalkeepers, Defenders,Midfielders, and Forwards. The goal is to select the best team walong with the most efficient team
# that can maximise points and also be within budget.

# Importing the necessary libraries
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Importing data sets
player_data = pd.read_csv('cleaned_players_shorter.csv')
player_data.head()
#Exporting specific data from the data set including player ID

#Apply a mimimun minutes played filter on the data set
player_data = player_data[player_data['minutes'] > 300]
print(player_data.head())

#games played
# Extracting the number of games played by each player
games_played = player_data['minutes']
games_played = games_played/90
player_data['games'] = games_played
player_data['games'] = player_data['games'].astype(int)
print(player_data.head())


# Goalkeepers
# Extracting the Goalkeepers from the data set
goalkeepers = player_data[player_data['Position'] == 'GK']
#print(goalkeepers.head())

# Defenders
# Extracting the Defenders from the data set
defenders = player_data[player_data['Position'] == 'DEF']
#print(defenders.head())

# Midfielders
# Extracting the Midfielders from the data set
midfielders = player_data[player_data['Position'] == 'MID']
#print(midfielders.head())

# Forwards
# Extracting the Forwards from the data set
forwards = player_data[player_data['Position'] == 'FWD']
#print(forwards.head())

# Assigning each club a unique number
club_names = player_data['Team'].unique()
club_dict = dict(zip(club_names, range(len(club_names))))
player_data['ClubID'] = player_data['Team'].map(club_dict)

#Now each club has a power ranking, the larger it is, the stronger the club
# Assigning each club a power ranking
club_power = player_data.groupby('Team')['Power ranking'].mean()
club_power_dict = dict(zip(club_power.index, club_power.values))
player_data['ClubPower'] = player_data['Team'].map(club_power_dict)
print (player_data.head())

# Points Constraint
#The objective of the game is to score the most points.
#Points are scored through goals, assists, clean sheets and saves.
#Points are deducted when a player receives a yellow card, red card or scores an own goal.
#Players are priced according to their points potential. 
#Logically, higher the price, higher the perceived points potential. 
#For example, Erling Haaland is priced at £14.0m, while someone like 
#Elliot Anderson is priced at £4.5m. 

#The points system is as follows:
#Goalkeepers :
#1 point for every 3 saves
#6 points for a clean sheet
#2 points for every goal scored
#3 points for every assist
#-1 point for every yellow card
#-3 points for every red card
#-2 points for every own goal  

# Defenders:
#6 points for a clean sheet
#2 points for every goal scored
#3 points for every assist
#-1 point for every yellow card
#-3 points for every red card
#-2 points for every own goal   

#Midfielders:
#5 points for every goal scored
#3 points for every assist
#-1 point for every yellow card
#-3 points for every red card
#-2 points for every own goal

#Forwards:
#4 points for every goal scored
#3 points for every assist
#-1 point for every yellow card
#-3 points for every red card
#-2 points for every own goal

#The objective is to maximize the points scored by the team.
#points weight

# Goalkeepers Weights
# Point Value Calculation:
saves = goalkeepers['Saves']
gk_clean_sheets = goalkeepers['clean_sheets']
gk_goals_conceded = goalkeepers['goals_conceded']
gk_yellow_cards = goalkeepers['yellow_cards']
gk_red_cards = goalkeepers['red_cards']
gk_games = goalkeepers['games']
#own_goals = goalkeepers['own_goals']

#Obtain average and convert to points per game for each goalkeeper
goalkeepers['Points'] = (saves/3) + gk_clean_sheets*6 + (gk_goals_conceded/2)*-1 + gk_yellow_cards*-1 + gk_red_cards*-3 + gk_games*2
keeper_points = dict(zip(goalkeepers['Player ID'], goalkeepers['Points']))
goalkeepers['Points'] = goalkeepers['Player ID'].map(keeper_points)
goalkeepers['Points'] = goalkeepers['Points'].fillna(0)
goalkeepers['Points'] = goalkeepers['Points'].astype(int)
goalkeepers['Points'] = goalkeepers['Points'].clip(lower=0)
print(goalkeepers.head())

# Defenders Weights
# Point Value Calculation:
def_clean_sheets = defenders['clean_sheets']
def_goals_scored = defenders['goals_scored']
def_assists = defenders['assists']
def_yellow_cards = defenders['yellow_cards']
def_red_cards = defenders['red_cards']
def_games = defenders['games']
#own_goals = defenders['own_goals']

#Obtain average and convert to points per game for each defender
defenders['Points'] = def_clean_sheets*6 + def_goals_scored*2 + def_assists*3 + def_yellow_cards*-1 + def_red_cards*-3 + def_games*2
defender_points = dict(zip(defenders['Player ID'], defenders['Points']))
defenders['Points'] = defenders['Player ID'].map(defender_points)
defenders['Points'] = defenders['Points'].fillna(0)
defenders['Points'] = defenders['Points'].astype(int)
defenders['Points'] = defenders['Points'].clip(lower=0)
print(defenders.head())

# Midfielders Weights
# Point Value Calculation:
mid_goals_scored = midfielders['goals_scored']
mid_assists = midfielders['assists']
mid_yellow_cards = midfielders['yellow_cards']
mid_red_cards = midfielders['red_cards']
mid_games = midfielders['games']
mid_clean_sheets = midfielders['clean_sheets']
#own_goals = midfielders['own_goals']

#Obtain average and convert to points per game for each midfielder
midfielders['Points'] = mid_goals_scored*5 + mid_assists*3 + mid_yellow_cards*-1 + mid_red_cards*-3 + mid_games*2 + mid_clean_sheets*1
midfielder_points = dict(zip(midfielders['Player ID'], midfielders['Points']))
midfielders['Points'] = midfielders['Player ID'].map(midfielder_points)
midfielders['Points'] = midfielders['Points'].fillna(0)
midfielders['Points'] = midfielders['Points'].astype(int)
midfielders['Points'] = midfielders['Points'].clip(lower=0)
print(midfielders.head())

# Forwards Weights
# Point Value Calculation:
fwd_goals_scored = forwards['goals_scored']
fwd_assists = forwards['assists']
fwd_yellow_cards = forwards['yellow_cards']
fwd_red_cards = forwards['red_cards']
fwd_games = forwards['games']

#Obtain average and convert to points per game for each forward
forwards['Points'] = fwd_goals_scored*4 + fwd_assists*3 + fwd_yellow_cards*-1 + fwd_red_cards*-3 + fwd_games*2
forward_points = dict(zip(forwards['Player ID'], forwards['Points']))
forwards['Points'] = forwards['Player ID'].map(forward_points)
forwards['Points'] = forwards['Points'].fillna(0)
forwards['Points'] = forwards['Points'].astype(int)
forwards['Points'] = forwards['Points'].clip(lower=0)
print(forwards.head())


# Using ICT index and the Teams Power rankings, try to get the Predicted Performace index that they would get
# A higher Power ranking means the team is stronger and would likely score more points

# Goalkeepers
# Goalkeepers Points Prediction
goalkeepers['Predicted Performace index'] = goalkeepers['Points'] + goalkeepers['ict_index']*goalkeepers['Power ranking']
#normalize
goalkeepers['Predicted Performace index'] = goalkeepers['Predicted Performace index']/ goalkeepers['Predicted Performace index'].min()
goalkeeper_points = dict(zip(goalkeepers['Player ID'], goalkeepers['Predicted Points']))
goalkeepers['Predicted Performace index'] = goalkeepers['Player ID'].map(goalkeeper_points)
goalkeepers['Predicted Performace index'] = goalkeepers['Predicted Performace index'].fillna(0)
goalkeepers['Predicted Performace index'] = goalkeepers['Predicted Performace index'].astype(int)
goalkeepers['Predicted Performace index'] = goalkeepers['Predicted Performace index'].clip(lower=0)
print(goalkeepers.head())

# Budget Constraint
#The total budget is £100m
#The objective is to use the budget to pick a squad of exactly 15 players (2 goalies, 5 defenders, 5 midfielders and 3 forwards.)
#From your squad of 15 players, you have to choose 11 playing players and 4 to sit on the bench. Of these 11 playing players, 
#there has to be exactly 1 goalie, at least 3 defenders, at least 2 midfielders and at least 1 forward.
# Points scored by players on the bench do not count towards your team’s total.
#You cannot pick the same player more than once and cannot have more than 3 players from the same team.

# Budget Constraint








