from collections import defaultdict
import numpy as np
import pandas as pd

# Function to simulate a single possession with EPV
def simulate_possession_with_epv(initial_state, transition_matrix, nba_df, poss):
    current_state = initial_state
    total_epv = 0  # Initialize total EPV for the possession
    
    while True:
        # Determine the next state based on the transition probabilities
        next_state = np.random.choice(transition_matrix.columns, p=transition_matrix.loc[current_state])
            
        if current_state in ['2pt shot', '3pt shot']:
            filtered_df = nba_df.loc[(nba_df['state'] == current_state) & 
                                     (nba_df['next_state'] == next_state) & 
                                     (nba_df['team'] == poss), ['EPV', 'shot_zone']]
            shot_zone_counts = filtered_df['shot_zone'].value_counts(normalize=True)
            
            if shot_zone_counts.empty:
                event_epv = 0
            else:
                chosen_shot_zone = np.random.choice(shot_zone_counts.index, p=shot_zone_counts.values)
                event_epv = filtered_df.loc[filtered_df['shot_zone'] == chosen_shot_zone, 'EPV'].mean()
        else:
            event_epv = nba_df.loc[(nba_df['state'] == current_state) & 
                                   (nba_df['next_state'] == next_state), 'EPV'].mean()
        
        total_epv += event_epv

        # If the next state is an absorbing state, end the possession
        if next_state in ['defensive rebound', 'turnover']:
            return next_state, total_epv
        
        # Update current state for the next iteration
        current_state = next_state


# Function to simulate games between two teams
def simulate_games(teamA, teamB, team_transition_matrices, nba_df, progress_callback=None):
    wins = defaultdict(int)
    scores = defaultdict(list)
    total_scores = []
    N = 1000
    for i in range(N):
        # Placeholder for team A and team B input
        poss = np.random.choice([teamA, teamB])
        track = defaultdict(float)
        total_epv_across_possessions = 0
        state = 'start of period'

        for _ in range(100):
            transition_matrix = team_transition_matrices[poss]
            next_state, end_epv = simulate_possession_with_epv(state, transition_matrix, nba_df, poss)
            track[poss] += end_epv
            
            if poss == teamA:
                poss = teamB
            else:
                poss = teamA

            total_epv_across_possessions += end_epv
            state = next_state
    
        if track[teamA] > track[teamB]:
            wins[teamA] += 1
        else:
            wins[teamB] += 1
            
        scores[teamA].append(track[teamA])
        scores[teamB].append(track[teamB])
        total_scores.append(total_epv_across_possessions)
    
        # Update progress bar
        if progress_callback:
            progress_callback(i + 1, N)
    
    return scores, wins, total_scores