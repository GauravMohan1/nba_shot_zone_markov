import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from simulate import simulate_games  # Assuming your simulation function is in simulate_game.py
import pickle 
import pandas as pd 
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

def main():
    # Load resources outside Streamlit functions
    filename_matrices = 'team_transition_matrices.pkl'
    filename_nba_data = 'nba_pbp_states.csv'
    
    team_transition_matrices = load_team_transition_matrices(filename_matrices)
    nba_df = load_nba_data(filename_nba_data)
    # Streamlit UI elements
    st.title('NBA Game Simulator')

    # Team selection dropdowns
    team_A = st.selectbox('Select Team A', nba_teams)
    team_B = st.selectbox('Select Team B', nba_teams)

    if st.button('Simulate'):
        # Create a progress bar
        progress_bar = st.progress(0)
        progress_text = st.empty()

        # Define a callback function to update the progress bar and text
        def progress_callback(current, total):
            progress = current / total
            progress_bar.progress(progress)
            progress_text.text(f'{progress * 100:.2f}% complete')

        #Run Simulation 
        scores, wins, total_scores = simulate_games(team_A, team_B, team_transition_matrices, nba_df, progress_callback)


        # Generate graphs
        # Plot line graph for scores
        plt.figure(figsize=(10, 6))
        for team in [team_A, team_B]:
            plt.plot(scores[team], label=team)
        plt.xlabel('Simulation Index')
        plt.ylabel('Score')
        plt.title('Scores across simulations')
        plt.axhline(np.mean(scores[team_A]), color='blue', linestyle='--', label='Mean score A')
        plt.axhline(np.median(scores[team_A]), color='blue', linestyle='-.', label='Median score A')
        plt.axhline(np.mean(scores[team_B]), color='orange', linestyle='--', label='Mean score B')
        plt.axhline(np.median(scores[team_B]), color='orange', linestyle='-.', label='Median score B')
        plt.legend()
        st.pyplot(plt)

        # Display win percentage
        st.write(f"Win percentage for {team_A}: {wins[team_A] / 10}%")
        st.write(f"Win percentage for {team_B}: {wins[team_B] / 10}%")

        # Display average and median total scores
        st.write(f"Average total score: {np.mean(total_scores)}")
        st.write(f"Median total score: {np.median(total_scores)}")


if __name__ == '__main__':
    main()