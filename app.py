import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd
from simulate import simulate_games  # Assuming your simulation function is in simulate_game.py
import plotly.graph_objects as go
import json
# List of NBA teams with their abbreviations
nba_teams = ['ATL', 'BOS', 'BKN', 'CHA', 'CHI', 'CLE', 'DAL', 'DEN', 'DET', 'GSW', 'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN', 'NOP', 'NYK', 'OKC', 'ORL', 'PHI', 'PHX', 'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS']

# Function to load team transition matrices from a pickle file
@st.cache(allow_output_mutation=True)
def load_team_transition_matrices(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

# Function to load NBA play-by-play states DataFrame from a CSV file
@st.cache(allow_output_mutation=True)
def load_nba_data(filename):
    return pd.read_csv(filename)

@st.cache(allow_output_mutation=True)
def load_pos_data(filename):
    with open(filename, 'r') as file:
        team_possessions = json.load(file)
        return team_possessions

def calculate_probabilities(simulated_scores, margin):
    # Initialize counters
    win_by_margin_count = 0
    
    # Loop through each simulation
    for score in simulated_scores:
        team1_score, team2_score = score
        point_diff = abs(team1_score - team2_score)
        
        # Check if the point difference matches the given margin
        if point_diff >= margin:
            win_by_margin_count += 1
    
    # Calculate probabilities
    total_simulations = len(simulated_scores)
    prob_win_by_margin = win_by_margin_count / total_simulations
    
    return prob_win_by_margin

def main():
    # Load resources outside Streamlit functions
    filename_matrices = 'team_transition_matrices.pkl'
    filename_nba_data = 'nba_pbp_states.csv'
    filename_pos = 'config_pos.json'

    team_transition_matrices = load_team_transition_matrices(filename_matrices)
    nba_df = load_nba_data(filename_nba_data)
    team_possessions = load_pos_data(filename_pos)
    
    # Streamlit UI elements
    st.title('NBA Game Simulator')

    # Team selection dropdowns
    team_A = st.selectbox('Select Team A', nba_teams)
    team_B = st.selectbox('Select Team B', nba_teams)

    N = st.number_input('Enter number of simulations', min_value=1, max_value=10000, value=7)
    average = (team_possessions[team_A] + team_possessions[team_B]) / 2
    
    # Round to the nearest even number
    num_pos = int(average // 2 * 2) if average % 2 != 0 else int(average)

    win_loss_margin = st.number_input('Enter Margin of Victory', min_value=1, max_value=10000, value=5)



    if st.button('Simulate'):
        # Create a progress bar and text placeholder
        progress_bar = st.progress(0)
        progress_text = st.empty()

        # Create placeholders for dynamic plots
        scores_plot_placeholder = st.empty()
        wins_plot_placeholder = st.empty()
        
        def update_plots(scores, wins):
            # Update scores plot
            scores_fig = go.Figure()
            if scores and wins:
                for team in [team_A, team_B]:
                    # Create x-axis values for the trace
                    x_values = [0] + list(range(1, len(scores[team]) + 1))
                    
                    # Create y-axis values for the trace
                    y_values = [0] + scores[team]

                    # Add trace to scores figure
                    scores_fig.add_trace(go.Scatter(
                        x=x_values,
                        y=y_values,
                        mode='lines',
                        name=f'{team} scores'
                    ))

            # Update layout of scores figure
            scores_fig.update_layout(
                title='Scores across simulations',
                xaxis_title='Simulation Index',
                yaxis_title='Score',
                xaxis=dict(range=[0, len(scores[team_A])]),  # Adjust x-axis range
                yaxis=dict(range=[0, 200])
            )

            scores_plot_placeholder.plotly_chart(scores_fig, use_container_width=True)
            # Update wins plot
            win_percentage_A = wins[team_A] / sum(wins.values()) * 100
            win_percentage_B = wins[team_B] / sum(wins.values()) * 100
            wins_fig = go.Figure(data=[
                go.Bar(name=team_A, x=[team_A], y=[win_percentage_A], marker_color='blue'),
                go.Bar(name=team_B, x=[team_B], y=[win_percentage_B], marker_color='orange')
            ])
            wins_fig.update_layout(
                title='Win Percentage of Teams',
                yaxis=dict(range=[0, 100]),
                yaxis_title='Win Percentage'
            )
            wins_plot_placeholder.plotly_chart(wins_fig, use_container_width=True)

        def progress_callback(current, total, cur_scores, cur_wins):
            progress = current / total
            progress_bar.progress(progress)
            progress_text.text(f'{progress * 100:.2f}% complete')
            update_plots(cur_scores, cur_wins)

        # Simulate games
        final_scores, total_scores = simulate_games(team_A, team_B, team_transition_matrices, nba_df, N, num_pos, progress_callback)

        teamA_scores = final_scores[team_A]
        teamB_scores = final_scores[team_B]
        scores_list = list(zip(teamA_scores, teamB_scores))

        # Calculate the point differentials
        point_differentials = [abs(a - b) for a, b in zip(final_scores[team_A], final_scores[team_B])]

        # Compute the average point differential
        avg_point_differential = round(sum(point_differentials) / len(point_differentials), 2)

        win_loss_probabilities = calculate_probabilities(scores_list, win_loss_margin)

        # Display average and median total scores
        st.write(f"Average total score: {np.mean(total_scores)}")
        st.write(f"Median total score: {np.median(total_scores)}")
        st.write(f"Absolute Avg Point Differential: {avg_point_differential}")
        st.write(f"Probability of margin of victory of {win_loss_margin} or more: {win_loss_probabilities}")
        st.write(f"Avg Points Per Game by {team_A}: {np.mean(final_scores[team_A])}")
        st.write(f"Avg Points Per Game by {team_B}: {np.mean(final_scores[team_B])}")


if __name__ == '__main__':
    main()