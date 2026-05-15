import streamlit as st
import numpy as np
import pygame
import os
import time

# --- STREAMLIT SETUP ---
st.set_page_config(page_title="Q-Learning Maze Solver")
st.title("🤖 Q-Learning Maze Solver")

# Force Pygame to be "headless" (no window)
os.environ["SDL_VIDEODRIVER"] = "dummy"
pygame.init()

# --- MAZE & LOGIC (Same as yours) ---
maze = [
    [0, 0, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 0],
    [1, 0, 1, 0, 1, 0],
    [1, 2, 0, 0, 0, 0],
    [1, 0, 1, 0, 1, 1]
]

CELL_SIZE = 60 # Reduced for web view
maze_rows, maze_cols = len(maze), len(maze[0])
q_values = np.zeros((maze_rows, maze_cols, 4))
actions = ['up', 'down', 'left', 'right']

# Setup rewards
rewards = np.zeros((maze_rows, maze_cols))
for r in range(maze_rows):
    for c in range(maze_cols):
        if maze[r][c] == 0: rewards[r][c] = -1
        elif maze[r][c] == 1: rewards[r][c] = -100
        elif maze[r][c] == 2: rewards[r][c] = 100

# Create a Pygame Surface (not a display window)
surface = pygame.Surface((maze_cols * CELL_SIZE, maze_rows * CELL_SIZE))

def get_maze_image(agent_pos=None):
    surface.fill((255, 255, 255)) # WHITE
    for i in range(maze_rows):
        for j in range(maze_cols):
            rect = pygame.Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE-2, CELL_SIZE-2)
            if maze[i][j] == 1: pygame.draw.rect(surface, (0,0,0), rect) # BLACK
            elif maze[i][j] == 2: pygame.draw.rect(surface, (0,255,0), rect) # GREEN
            else: pygame.draw.rect(surface, (200,200,200), rect, 1)
    
    if agent_pos:
        center = (agent_pos[1]*CELL_SIZE + CELL_SIZE//2, agent_pos[0]*CELL_SIZE + CELL_SIZE//2)
        pygame.draw.circle(surface, (255,0,0), center, CELL_SIZE//3) # RED

    # Convert surface to image array
    view = pygame.surfarray.array3d(surface)
    return np.transpose(view, (1, 0, 2)) # Transpose for Streamlit image format

# --- Q-LEARNING HELPERS ---
def is_terminal_state(r, c): return maze[r][c] == 2

def get_next_action(r, c, epsilon):
    if np.random.random() < epsilon: return np.random.randint(4)
    return np.argmax(q_values[r, c])

def get_next_location(r, c, action_idx):
    nr, nc = r, c
    if actions[action_idx] == 'up' and r > 0 and maze[r-1][c] != 1: nr -= 1
    elif actions[action_idx] == 'down' and r < maze_rows-1 and maze[r+1][c] != 1: nr += 1
    elif actions[action_idx] == 'left' and c > 0 and maze[r][c-1] != 1: nc -= 1
    elif actions[action_idx] == 'right' and c < maze_cols-1 and maze[r][c+1] != 1: nc += 1
    return nr, nc

# --- STREAMLIT UI ---
if st.button("🚀 Start Training"):
    placeholder = st.empty() # Container to update the maze image
    progress_bar = st.progress(0)
    
    learning_rate = 0.9
    discount_factor = 0.95
    epsilon = 1.0
    epochs = 400 # Keep it lower for web demo speed

    for episode in range(epochs):
        ar, ac = 0, 0
        while not is_terminal_state(ar, ac):
            action = get_next_action(ar, ac, epsilon)
            nr, nc = get_next_location(ar, ac, action)
            
            # Q-Update Logic
            td = rewards[nr, nc] + (discount_factor * np.max(q_values[nr, nc])) - q_values[ar, ac, action]
            q_values[ar, ac, action] += learning_rate * td
            ar, ac = nr, nc
        
        epsilon = max(0.01, epsilon * 0.99)
        
        # Update visualization every 20 episodes
        if episode % 20 == 0:
            img = get_maze_image((ar, ac))
            placeholder.image(img, caption=f"Training Episode: {episode}")
            progress_bar.progress(episode / epochs)

    st.success("Training Complete!")
    
    # Show Shortest Path
    st.subheader("Final Shortest Path")
    path_placeholder = st.empty()
    cr, cc = 0, 0
    while not is_terminal_state(cr, cc):
        path_placeholder.image(get_maze_image((cr, cc)))
        action = np.argmax(q_values[cr, cc])
        cr, cc = get_next_location(cr, cc, action)
        time.sleep(0.3)
    path_placeholder.image(get_maze_image((cr, cc)))
