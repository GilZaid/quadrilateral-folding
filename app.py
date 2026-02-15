import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ===== FUNCTION DEFINITIONS =====

def fold(v0, v1, v2, v3):
    """Apply cyclic folding to a given quadrilateral and return the new set
    of vertices.
    """
    v0_folded = (np.conj(v0 - v1) * (v3 - v1)) / np.conj(v3 - v1) + v1
    new_v0 = v1
    new_v1 = v2
    new_v2 = v3
    new_v3 = v0_folded
    return new_v0, new_v1, new_v2, new_v3

def plot_orbit_to_image(mu, nu, iters, plotsize, pointsize=5):
    """Plot orbit and return as image."""
    v0 = -np.sqrt(1 + mu*nu - mu - nu) + 1j*mu
    v1 = np.sqrt(1 + mu*nu - mu - nu) + 1j*nu
    v2 = v1.conjugate()
    v3 = v0.conjugate()

    all_points = [v0, v1, v2, v3]
    for i in range(iters):
        v0, v1, v2, v3 = fold(v0, v1, v2, v3)
        all_points.extend([v0, v1, v2, v3])

    x = [z.real for z in all_points]
    y = [z.imag for z in all_points]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(x, y, color='black', s=pointsize, alpha=0.6)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-plotsize, plotsize)
    ax.set_ylim(-plotsize, plotsize)
    ax.set_xlabel('Real')
    ax.set_ylabel('Imaginary')
    ax.set_title(f'Orbit over {iters} iterations (μ={mu}, ν={nu})')
    
    return fig

def animate_folding(mu, nu, iters, duration, plotsize=3, pointsize=2, 
                    orbit=False, iters_orbit=1000, alpha_orbit=0.3):
    """Animate the folding map applied repeatedly."""
    
    # Compute orbit for backdrop
    if orbit:
        v0_orbit = -np.sqrt(1 + mu*nu - mu - nu) + 1j*mu
        v1_orbit = np.sqrt(1 + mu*nu - mu - nu) + 1j*nu
        v2_orbit = v1_orbit.conjugate()
        v3_orbit = v0_orbit.conjugate()

        orbit_points = [v0_orbit, v1_orbit, v2_orbit, v3_orbit]
        for i in range(iters_orbit):
            v0_orbit, v1_orbit, v2_orbit, v3_orbit = fold(v0_orbit, v1_orbit, v2_orbit, v3_orbit)
            orbit_points.extend([v0_orbit, v1_orbit, v2_orbit, v3_orbit])

        orbit_x = [z.real for z in orbit_points]
        orbit_y = [z.imag for z in orbit_points]

    # Initialize vertices
    v0 = -np.sqrt(1 + mu*nu - mu - nu) + 1j*mu
    v1 = np.sqrt(1 + mu*nu - mu - nu) + 1j*nu
    v2 = v1.conjugate()
    v3 = v0.conjugate()

    # Generate all frames
    frames = [(v0, v1, v2, v3)]
    for i in range(iters):
        v0, v1, v2, v3 = fold(v0, v1, v2, v3)
        frames.append((v0, v1, v2, v3))

    # Set up the figure
    fig, ax = plt.subplots(figsize=(8, 8))

    def update(frame_num):
        ax.clear()

        # Draw orbit backdrop
        if orbit:
            ax.scatter(orbit_x, orbit_y, color='gray', s=pointsize, alpha=alpha_orbit)

        # Draw current quadrilateral
        v0, v1, v2, v3 = frames[frame_num]
        vertices = [v0, v1, v2, v3]
        x = [z.real for z in vertices] + [v0.real]
        y = [z.imag for z in vertices] + [v0.imag]

        ax.fill(x, y, color='lightgray', alpha=0.5)
        ax.plot(x, y, 'k-', linewidth=1)
        ax.scatter([z.real for z in vertices], [z.imag for z in vertices],
                   color='black', s=20, zorder=5)

        ax.axhline(0, color='k', linewidth=0.5)
        ax.axvline(0, color='k', linewidth=0.5)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-plotsize, plotsize)
        ax.set_ylim(-plotsize, plotsize)
        ax.set_title(f'Iteration {frame_num}')

    anim = FuncAnimation(fig, update, frames=len(frames), interval=duration, repeat=True)
    plt.close()
    return anim.to_jshtml()

# ===== STREAMLIT UI =====

st.set_page_config(page_title="Iterated Folding Visualizer", layout="centered")

# Custom CSS for black and white theme
st.markdown("""
    <style>
    .stApp {
        background-color: white;
    }
    h1 {
        color: black;
        text-align: center;
        font-weight: 400;
    }
    /* Make slider thumbs (circles) black */
    .stSlider [role="slider"] {
        background-color: black !important;
    }
    /* Make slider active track black */
    .stSlider [data-baseweb="slider"] > div > div {
        background-color: black !important;
    }
    /* Make checkbox checkmark black */
    .stCheckbox input[type="checkbox"]:checked + div {
        background-color: black !important;
        border-color: black !important;
    }
    /* Make radio button selected black */
    .stRadio [role="radiogroup"] label[data-checked="true"] div:first-child {
        background-color: black !important;
        border-color: black !important;
    }
    /* Make buttons black */
    .stButton > button {
        background-color: black;
        color: white;
        border: 1px solid black;
    }
    .stButton > button:hover {
        background-color: #333;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("Iterated Folding Visualizer")

# Mode selection at top
mode = st.radio("", ["Plot Orbit", "Animate Folding"], horizontal=True, label_visibility="collapsed")

# All parameters at the top
col1, col2, col3 = st.columns(3)

with col1:
    mu = st.slider("μ (mu)", 0.0, 1.0, 0.5, 0.01)
    
with col2:
    nu = st.slider("ν (nu)", 0.0, 1.0, 0.5, 0.01)
    
with col3:
    plotsize = st.slider("Plot Size", 1, 20, 3, 1)

if mode == "Plot Orbit":
    col1, col2, col3 = st.columns(3)
    
    with col1:
        iters = st.slider("Iterations", 10, 5000, 1000, 10)
    
    with col2:
        pointsize = st.slider("Point Size", 1, 10, 5, 1)
    
    with col3:
        st.write("")  # spacer

    generate = st.button("Generate Orbit Plot", type="primary", use_container_width=True)
    
    if generate:
        with st.spinner("Generating orbit..."):
            fig = plot_orbit_to_image(mu, nu, iters, plotsize, pointsize)
            st.pyplot(fig, use_container_width=True)
            plt.close()

else:  # Animate Folding
    col1, col2, col3 = st.columns(3)
    
    with col1:
        iters = st.slider("Animation Iterations", 1, 100, 20, 1)
    
    with col2:
        duration = st.slider("Frame Duration (ms)", 50, 1000, 200, 50)
    
    with col3:
        pointsize = st.slider("Point Size", 1, 10, 2, 1)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        orbit = st.checkbox("Show Orbit Background", value=False)
    
    with col2:
        iters_orbit = st.slider("Orbit Iterations", 100, 5000, 1000, 100) if orbit else 1000
    
    with col3:
        alpha_orbit = st.slider("Orbit Transparency", 0.0, 1.0, 0.3, 0.05) if orbit else 0.3

    generate = st.button("Generate Animation", type="primary", use_container_width=True)
    
    if generate:
        with st.spinner("Generating animation..."):
            html_anim = animate_folding(mu, nu, iters, duration, plotsize, 
                                       pointsize, orbit, iters_orbit, 
                                       alpha_orbit)
            st.components.v1.html(html_anim, height=650, scrolling=False)
