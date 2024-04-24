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
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from pulp import (
    LpMaximize, LpProblem, LpVariable, lpSum, LpStatus, value, LpMinimize, LpInteger,
    LpBinary, LpContinuous, LpAffineExpression, LpConstraint, LpConstraintEQ, LpConstraintGE, LpConstraintLE
)

# Importing data sets
player_data = pd.read_csv('cleaned_players_shorter.csv')
player_data.head()
#Exporting specific data from the data set including player ID

#Apply a mimimun minutes played filter on the data set
player_data = player_data[player_data['minutes'] > 300]
#print(player_data.head())

#games played
# Extracting the number of games played by each player
games_played = player_data['minutes']
games_played = games_played/90
player_data['games'] = games_played
player_data['games'] = player_data['games'].astype(int)
#print(player_data.head())

player_data = pd.DataFrame(player_data)

#Generate a club id to add to the data
team_coes = {}
for i, team in enumerate(player_data['Team'].unique()):
    team_coes[team] = i
player_data['ClubID'] = player_data['Team'].map(team_coes)


#Collect each club and their power ranking
club_names = player_data['Team'].unique()
club_dict = dict(zip(club_names, range(len(club_names))))
club_power = player_data.groupby('Team')['Power ranking'].mean()
club_power_dict = dict(zip(club_power.index, club_power.values)) #Power ranking of each club

#Player team changes due to the transfer window
team_changes = {
    'Mason Mount': 'Manchester United',
    'James Maddison': 'Tottenham',
    'Declan Rice': 'Arsenal',
    'Kai Havertz': 'Arsenal',
    'Youri Tielemans': 'Aston Villa',
    'João Pedro Junqueira de Jesus': 'Brighton',
    'Robert Sánchez': 'Chelsea',
    'Ashley Young': 'Everton',
    'Alexis Mac Allister': 'Liverpool',
    'Harvey Barnes': 'Newcastle',
    'David Raya Martin': 'Arsenal',
    'Christian Eriksen': 'Manchester United'
   # 'Anthony Elanga': 'Nottingham'
}

#Extract the payer name and get their info from the data set
# then update the Team and Power ranking

for player, team in team_changes.items():
    player_data.loc[player_data['Name'] == player, 'Team'] = team
    player_data.loc[player_data['Name'] == player, 'Power ranking'] = club_power_dict[team]

#Upddate player prices
new_prices = {
    'David Raya Martin': 52,
    'Erling Haaland': 140,
    'Harry Kane': 125,
    'Mohamed Salah': 125,
    'Kevin De Bruyne': 105,
    'Marcus Rashford': 90,
    'Son Heung-min': 90,
    'Heung-Min Son': 90,
    'Bruno Miguel Borges Fernandes': 85,
    'Bukayo Saka': 85,
    'Martin Ødegaard': 85,
    'Trent Alexander-Arnold': 80,
    'Diogo Jota': 80,
    'Gabriel Martinelli': 80,
    'Ollie Watkins': 80,
    'Callum Wilson': 80,
    'Darwin Nuñez': 75,
    'Phil Foden': 75,
    'Jack Grealish': 75,
    'Kai Havertz': 75,
    'Luis Diaz': 75,
    'James Maddison': 75,
    'Aleksandar Mitrović': 75,
    'Jarrod Bowen': 70,
    'Dejan Kulusevski': 70,
    'Mason Mount': 70,
    'Richarlison': 70,
    'Jadon Sancho': 70,
    'Raheem Sterling': 70,
    'Leandro Trossard': 70,
    'Miguel Almirón Rejala': 65,
    'Harvey Barnes': 65,
    'Bernardo Veiga de Carvalho e Silva': 65,
    'Moussa Diaby': 65,
    'Eberechi Eze': 65,
    'Pascal Gross': 65,
    'Julian Alvarez': 65,
    'Solly March': 65,
    'Kaoru Mitoma': 65,
    'Kieran Trippier': 65,
    'Joao Cancelo': 60,
    'Danny Ings': 60,
    'Lucas Paqueta': 60,
    'Alexis Mac Allister': 60,
    'Virgil Van Dijk': 60,
    'Yoane Wissa': 60,
    'Reece James': 55,
    'Benjamin Chilwell': 55,
    'Alisson Ramses Becker': 55,
    'Ederson Santana de Moraes': 55,
    'Nick Pope': 55,
    'Eddie Nketiah': 55,
    'Ruben Gato Alves Dias': 55,
    'John Stones': 55,
    'Luke Shaw': 55,
    'Aaron Ramsdale': 50,
    'Pervis Estupiñan': 50,
    'William Saliba': 50,
    'Fabian Schär': 50,
    'Gabriel dos Santos Magalhães': 50,
    'Robert Sanchez': 45,
    'Jordan Pickford': 45,
    'Sven Botman': 45,
    'Tyrone Mings': 45
}

# Remove players from the data set who are currently not in the league
players_to_remove = ['Denis Zakaria', 'João Félix Sequeira ', 'Mason Holgate', 'Jadon Sancho','Fábio Freitas Gouveia Carvalho','Anthony Elanga'\
                     'Sékou Mara', 'Kepa Arrizabalaga', 'Matías Viña', 'Pontus Jansson', 'Ivan Perišić', 'Harry Kane']
player_data = player_data[~player_data['Name'].isin(players_to_remove)]


#Replace player prices with new prices
for player, price in new_prices.items():
    player_data.loc[player_data['Name'] == player, 'cost'] = price

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
#print (player_data.head())

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
#print(goalkeepers.head())

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
#print(defenders.head())

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
#print(midfielders.head())

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
#print(forwards.head())


# Using ICT index and the Teams Power rankings, try to get the Predicted Performace index that they would get
# A higher Power ranking means the team is stronger and would likely score more points

# Goalkeepers
# Goalkeepers Performance index
goalkeepers['Predicted Performace index'] = goalkeepers['Points'] + goalkeepers['ict_index']*goalkeepers['Power ranking']
#normalize
goalkeepers['Predicted Performace index'] = goalkeepers['Predicted Performace index']/ goalkeepers['Predicted Performace index'].min()
goalkeeper_points = dict(zip(goalkeepers['Player ID'], goalkeepers['Predicted Performace index']))
goalkeepers['Predicted Performace index'] = goalkeepers['Player ID'].map(goalkeeper_points)
goalkeepers['Predicted Performace index'] = goalkeepers['Predicted Performace index'].fillna(0)
goalkeepers['Predicted Performace index'] = goalkeepers['Predicted Performace index'].astype(int)
goalkeepers['Predicted Performace index'] = goalkeepers['Predicted Performace index'].clip(lower=0)
#print(goalkeepers.head())

# Defenders
# Defenders Performance index
defenders['Predicted Performace index'] = defenders['Points'] + defenders['ict_index']*defenders['Power ranking']
#normalize
defenders['Predicted Performace index'] = defenders['Predicted Performace index']/ defenders['Predicted Performace index'].min()
defender_points = dict(zip(defenders['Player ID'], defenders['Predicted Performace index']))
defenders['Predicted Performace index'] = defenders['Player ID'].map(defender_points)
defenders['Predicted Performace index'] = defenders['Predicted Performace index'].fillna(0)
defenders['Predicted Performace index'] = defenders['Predicted Performace index'].astype(int)
defenders['Predicted Performace index'] = defenders['Predicted Performace index'].clip(lower=0)
#print(defenders.head())

# Midfielders
# Midfielders Performance index
midfielders['Predicted Performace index'] = midfielders['Points'] + midfielders['ict_index']*midfielders['Power ranking']
#normalize
midfielders['Predicted Performace index'] = midfielders['Predicted Performace index']/ midfielders['Predicted Performace index'].min()
midfielder_points = dict(zip(midfielders['Player ID'], midfielders['Predicted Performace index']))
midfielders['Predicted Performace index'] = midfielders['Player ID'].map(midfielder_points)
midfielders['Predicted Performace index'] = midfielders['Predicted Performace index'].fillna(0)
midfielders['Predicted Performace index'] = midfielders['Predicted Performace index'].astype(int)
midfielders['Predicted Performace index'] = midfielders['Predicted Performace index'].clip(lower=0)
#print(midfielders.head())

# Forwards
# Forwards Performance index
forwards['Predicted Performace index'] = forwards['Points'] + forwards['ict_index']*forwards['Power ranking']
#normalize
forwards['Predicted Performace index'] = forwards['Predicted Performace index']/ forwards['Predicted Performace index'].min()
forward_points = dict(zip(forwards['Player ID'], forwards['Predicted Performace index']))
forwards['Predicted Performace index'] = forwards['Player ID'].map(forward_points)
forwards['Predicted Performace index'] = forwards['Predicted Performace index'].fillna(0)
forwards['Predicted Performace index'] = forwards['Predicted Performace index'].astype(int)
forwards['Predicted Performace index'] = forwards['Predicted Performace index'].clip(lower=0)
#print(forwards.head())

#add each players predicted performance index to the player data
player_data['Predicted Performace index'] = 0
player_data.loc[player_data['Position'] == 'GK', 'Predicted Performace index'] = goalkeepers['Predicted Performace index']
player_data.loc[player_data['Position'] == 'DEF', 'Predicted Performace index'] = defenders['Predicted Performace index']
player_data.loc[player_data['Position'] == 'MID', 'Predicted Performace index'] = midfielders['Predicted Performace index']
player_data.loc[player_data['Position'] == 'FWD', 'Predicted Performace index'] = forwards['Predicted Performace index']

#CONSTRAINED OPTIMIZATION

#The objective of the game is to score the most points.
#Points are scored through goals, assists, clean sheets and saves.

# The budget is 1000
budget = 1000

# The number of players to select
num_players = 15

# The number of goalkeepers to select
num_goalkeepers = 2

# The number of defenders to select
num_defenders = 5

# The number of midfielders to select
num_midfielders = 5

# The number of forwards to select
num_forwards = 3

# The number of players from the same team/club
max_players_from_same_team = 3

#Cost efficiency
# factor in team strength to cost efficiency. Meaning the stronger the team, the more efficient the cost may be
# Adjust the cost efficiency calculation to factor in club power
# Alpha factor for cost penalization
alpha = 0.1  # Adjust this based on how sensitive you want to be towards cost

# Calculate Cost Efficiency
player_data['Cost Efficiency'] = (player_data['Predicted Performace index'] * (player_data['Power ranking'] + 1)) / (player_data['cost'] ** alpha)
# The optimization function

def optimize_team(player_data, budget, num_players, num_goalkeepers, num_defenders, num_midfielders, num_forwards, max_players_from_same_team):
    # Create a binary variable for each player
    players = player_data['Player ID'].values
    player_vars = LpVariable.dicts('Players', players, 0, 1, LpBinary)

    # Create the optimization model
    model = LpProblem('OptimizeTeam', LpMaximize)

    # Objective function maximizing cost-efficiency
    model += lpSum([player_vars[player] * player_data.loc[player_data['Player ID'] == player, 'Cost Efficiency'].values[0] for player in players])

    # Number of players constraint
    model += lpSum([player_vars[player] for player in players]) == num_players

    # Budegt constraint
    model += lpSum([player_vars[player] * player_data.loc[player_data['Player ID'] == player, 'cost'].values[0] for player in players]) <= budget

    # Position-specific constraints
    goalkeepers = player_data[player_data['Position'] == 'GK']['Player ID'].values
    defenders = player_data[player_data['Position'] == 'DEF']['Player ID'].values
    midfielders = player_data[player_data['Position'] == 'MID']['Player ID'].values
    forwards = player_data[player_data['Position'] == 'FWD']['Player ID'].values

    model += lpSum([player_vars[player] for player in goalkeepers]) == num_goalkeepers
    model += lpSum([player_vars[player] for player in defenders]) == num_defenders
    model += lpSum([player_vars[player] for player in midfielders]) == num_midfielders
    model += lpSum([player_vars[player] for player in forwards]) == num_forwards

    # Maximum number of players from the same team constraint
    clubs = player_data['ClubID'].unique()
    for club in clubs:
        club_players = player_data[player_data['ClubID'] == club]['Player ID'].values
        model += lpSum([player_vars[player] for player in club_players]) <= max_players_from_same_team

    # Solve the optimization problem
    model.solve()

    # Extract the selected players
    selected_goalkeepers = [player for player in goalkeepers if player_vars[player].varValue == 1]
    selected_defenders = [player for player in defenders if player_vars[player].varValue == 1]
    selected_midfielders = [player for player in midfielders if player_vars[player].varValue == 1]
    selected_forwards = [player for player in forwards if player_vars[player].varValue == 1]

    return selected_goalkeepers, selected_defenders, selected_midfielders, selected_forwards

'''
def optimize_team(player_data, budget, num_players, num_goalkeepers, num_defenders, num_midfielders, num_forwards, max_players_from_same_team):
    # Create a binary variable for each player
    players = player_data['Player ID'].values
    player_vars = LpVariable.dicts('Players', players, 0, 1, LpBinary)

    # Create a binary variable for each goalkeeper
    goalkeepers = player_data[player_data['Position'] == 'GK']['Player ID'].values
    goalkeeper_vars = LpVariable.dicts('Goalkeepers', goalkeepers, 0, 1, LpBinary)

    # Create a binary variable for each defender
    defenders = player_data[player_data['Position'] == 'DEF']['Player ID'].values
    defender_vars = LpVariable.dicts('Defenders', defenders, 0, 1, LpBinary)

    # Create a binary variable for each midfielder
    midfielders = player_data[player_data['Position'] == 'MID']['Player ID'].values
    midfielder_vars = LpVariable.dicts('Midfielders', midfielders, 0, 1, LpBinary)

    # Create a binary variable for each forward
    forwards = player_data[player_data['Position'] == 'FWD']['Player ID'].values
    forward_vars = LpVariable.dicts('Forwards', forwards, 0, 1, LpBinary)

    # Create a binary variable for each club
    clubs = player_data['ClubID'].unique()
    club_vars = LpVariable.dicts('Clubs', clubs, 0, 1, LpBinary)

    # Create a binary variable for each player from the same team
    players_same_team = player_data['ClubID'].values
    player_vars_same_team = LpVariable.dicts('Players_Same_Team', players_same_team, 0, 1, LpBinary)

    # Create a binary variable for each premium player (cost > 80) and (position-specific performance index > threshold)
    premium_players = player_data[(player_data['cost'] > 80) & (player_data['Predicted Performace index'] > 10) & (player_data['Power ranking'] > 90)]['Player ID'].values
    premium_player_vars = LpVariable.dicts('Premium_Players', premium_players, 0, 1, LpBinary)
    print ('Premium Players:', premium_players )
    # Create the optimization model
    # start by choosing the best players

    model = LpProblem('OptimizeTeam', LpMaximize)

   # Ensure at least one premium player is selected
    model += lpSum([premium_player_vars[player] for player in premium_players if player in premium_player_vars]) <= 1

    # Objective function maximizing cost-efficiency
    model += lpSum([player_vars[player] * player_data.loc[player_data['Player ID'] == player, 'Cost Efficiency'].values[0] for player in players])
    
    # Number of players constraint
    model += lpSum([player_vars[player] for player in players]) == num_players

    # Number of goalkeepers constraint
    model += lpSum([goalkeeper_vars[player] for player in goalkeepers]) == num_goalkeepers

    # Number of defenders constraint
    model += lpSum([defender_vars[player] for player in defenders]) == num_defenders

    # Number of midfielders constraint
    model += lpSum([midfielder_vars[player] for player in midfielders]) == num_midfielders

    # Number of forwards constraint
    model += lpSum([forward_vars[player] for player in forwards]) == num_forwards

    # At least one premium player constraint
    # Filter premium players based on cost and position-specific performance index

    # Solve the optimization problem
    model.solve()

    # Extract the selected players

    selected_goalkeepers = [player for player in goalkeepers if goalkeeper_vars[player].varValue == 1]
    selected_defenders = [player for player in defenders if defender_vars[player].varValue == 1]
    selected_midfielders = [player for player in midfielders if midfielder_vars[player].varValue == 1]
    selected_forwards = [player for player in forwards if forward_vars[player].varValue == 1]

    return selected_goalkeepers, selected_defenders, selected_midfielders, selected_forwards
'''

# Optimize the team
selected_goalkeepers, selected_defenders, selected_midfielders, selected_forwards = optimize_team(player_data, budget, num_players, num_goalkeepers, num_defenders, num_midfielders, num_forwards, max_players_from_same_team)

# Print the selected players, their team and their cost

print (f'Budget: £{budget}m\n')
# Goalkeepers
print('Goalkeepers:')
for player in selected_goalkeepers:
    player_info = player_data[player_data['Player ID'] == player]
    print(f'{player_info["Name"].values[0]} - {player_info["Team"].values[0]} - £{player_info["cost"].values[0]}m')

# Defenders
print('\nDefenders:')
for player in selected_defenders:
    player_info = player_data[player_data['Player ID'] == player]
    print(f'{player_info["Name"].values[0]} - {player_info["Team"].values[0]} - £{player_info["cost"].values[0]}m')

# Midfielders
print('\nMidfielders:')
for player in selected_midfielders:
    player_info = player_data[player_data['Player ID'] == player]
    print(f'{player_info["Name"].values[0]} - {player_info["Team"].values[0]} - £{player_info["cost"].values[0]}m')

# Forwards
print('\nForwards:')
for player in selected_forwards:
    player_info = player_data[player_data['Player ID'] == player]
    print(f'{player_info["Name"].values[0]} - {player_info["Team"].values[0]} - £{player_info["cost"].values[0]}m')

print(f'\nTotal spent: £{player_data.loc[player_data["Player ID"].isin(selected_goalkeepers + selected_defenders + selected_midfielders + selected_forwards), "cost"].sum()}m')
#export player data to pdf or txt
#player_data.to_csv('selected_players.csv', index=False)
