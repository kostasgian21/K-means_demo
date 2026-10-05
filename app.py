import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import plotly.express as px
import warnings

# Suppress convergence warnings for early iterations
warnings.filterwarnings("ignore")

st.title("Interactive K-Means Educator")

st.markdown("""
**How K-Means works:** 
1. Randomly places 'K' centroids. 
2. Assigns points to the nearest centroid. 
3. Moves the centroid to the middle of those points. 
*Use the Iteration slider below to watch this happen step-by-step!*
""")

# --- STATE MANAGEMENT (Restart & Data) ---
if 'data' not in st.session_state:
    # Generate some random starter blobs
    np.random.seed(42)
    x = np.concatenate([np.random.normal(0, 1, 10), np.random.normal(5, 1.5, 10), np.random.normal(2, 1, 10)])
    y = np.concatenate([np.random.normal(0, 1, 10), np.random.normal(5, 1, 10), np.random.normal(-2, 1, 10)])
    st.session_state.data = pd.DataFrame({'X': x, 'Y': y})

def restart_app():
    del st.session_state.data

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Parameters")
k = st.sidebar.number_input("Number of Clusters (K):", min_value=1, max_value=5, value=3)

# The "GIF" automation component
iteration = st.sidebar.slider("Algorithm Iteration (Drag to Animate):", min_value=1, max_value=10, value=1)

st.sidebar.button("Reset Dashboard", on_click=restart_app)

st.sidebar.header("Add Data Points")
st.sidebar.write("Add or edit rows to see how outliers change the clusters.")
# Interactive data editor allows users to add/delete points
edited_data = st.sidebar.data_editor(st.session_state.data, num_rows="dynamic", hide_index=True)
st.session_state.data = edited_data

# --- K-MEANS COMPUTATION ---
df = st.session_state.data.copy()

if len(df) >= k:
    # We restrict max_iter to simulate the step-by-step animation
    kmeans = KMeans(n_clusters=k, max_iter=iteration, init='random', n_init=1, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df[['X', 'Y']]).astype(str)
    centroids = pd.DataFrame(kmeans.cluster_centers_, columns=['X', 'Y'])
    
    # --- PLOTTING ---
    fig = px.scatter(df, x='X', y='Y', color='Cluster', size_max=10, 
                     title=f"K-Means at Iteration {iteration}",
                     color_discrete_sequence=px.colors.qualitative.Set1)
    
    # Add centroids as large black X's
    fig.add_scatter(x=centroids['X'], y=centroids['Y'], mode='markers', 
                    marker=dict(color='black', size=15, symbol='x', line=dict(width=3)),
                    name='Centroids')
    
    fig.update_traces(marker=dict(size=12))
    fig.update_layout(xaxis_range=[-5, 10], yaxis_range=[-5, 10]) # Fixed axes to see movement clearly
    
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning(f"Please provide at least {k} data points.")