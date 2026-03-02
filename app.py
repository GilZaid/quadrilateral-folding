import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# =========================
# Folding Functions
# =========================

def fold(v0, v1, v2, v3):
    """Apply cyclic folding to a given quadrilateral."""
    v0_folded = (np.conj(v0 - v1) * (v3 - v1)) / np.conj(v3 - v1) + v1
    new_v0 = v1
    new_v1 = v2
    new_v2 = v3
    new_v3 = v0_folded
    return new_v0, new_v1, new_v2, new_v3

def fold_centered(v0, v1, v2, v3):
    """Apply cyclic folding, then recenter vertices at origin."""
    v0, v1, v2, v3 = fold(v0, v1, v2, v3)

    center = (v0 + v1 + v2 + v3) / 4

    v0 -= center
    v1 -= center
    v2 -= center
    v3 -= center

    return v0, v1, v2, v3

# =========================
# Orbit Plot (unchanged)
# =========================

def plot_orbit_to_image(mu, nu, iters, plotsize, pointsize=5):
    v0 = -np.sqrt(1 + mu * nu - mu - nu) + 1j * mu
    v1 = np.sqrt(1 + mu * nu - mu - nu) + 1j * nu
    v2 = v1.conjugate()
    v3 = v0.conjugate()

    all_points = [v0, v1, v2, v3]

    for _ in range(iters):
        v0, v1, v2, v3 = fold(v0, v1, v2, v3)
        all_points.extend([v0, v1, v2, v3])

    x = [z.real for z in all_points]
    y = [z.imag for z in all_points]

    fig, ax = plt.subplots(figsize=(7, 7))
    fig.subplots_adjust(top=0.92)

    ax.scatter(x, y, color="black", s=pointsize, alpha=0.6)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.axvline(0, color="k", linewidth=0.5)
    ax.grid(True, alpha=0.3)

    ax.set_xlim(-plotsize, plotsize)
    ax.set_ylim(-plotsize, plotsize)

    ax.set_xlabel("Real")
    ax.set_ylabel("Imaginary")
    ax.set_title(f"Orbit over {iters} iterations (μ={mu}, ν={nu})", pad=12)

    return fig

# =========================
# Animation (unchanged)
# =========================

def animate_folding(
    mu,
    nu,
    iters,
    duration,
    plotsize=3,
    pointsize=2,
    orbit=False,
    iters_orbit=1000,
    alpha_orbit=0.3,
):
    if orbit:
        v0_orbit = -np.sqrt(1 + mu * nu - mu - nu) + 1j * mu
        v1_orbit = np.sqrt(1 + mu * nu - mu - nu) + 1j * nu
        v2_orbit = v1_orbit.conjugate()
        v3_orbit = v0_orbit.conjugate()

        orbit_points = [v0_orbit, v1_orbit, v2_orbit, v3_orbit]

        for _ in range(iters_orbit):
            v0_orbit, v1_orbit, v2_orbit, v3_orbit = fold(
                v0_orbit, v1_orbit, v2_orbit, v3_orbit
            )
            orbit_points.extend([v0_orbit, v1_orbit, v2_orbit, v3_orbit])

        orbit_x = [z.real for z in orbit_points]
        orbit_y = [z.imag for z in orbit_points]

    v0 = -np.sqrt(1 + mu * nu - mu - nu) + 1j * mu
    v1 = np.sqrt(1 + mu * nu - mu - nu) + 1j * nu
    v2 = v1.conjugate()
    v3 = v0.conjugate()

    frames = [(v0, v1, v2, v3)]

    for _ in range(iters):
        v0, v1, v2, v3 = fold(v0, v1, v2, v3)
        frames.append((v0, v1, v2, v3))

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    fig.subplots_adjust(top=0.92)

    def update(frame_num):
        ax.clear()

        if orbit:
            ax.scatter(
                orbit_x,
                orbit_y,
                color="gray",
                s=pointsize,
                alpha=alpha_orbit,
            )

        v0, v1, v2, v3 = frames[frame_num]
        vertices = [v0, v1, v2, v3]

        x = [z.real for z in vertices] + [v0.real]
        y = [z.imag for z in vertices] + [v0.imag]

        ax.fill(x, y, color="lightgray", alpha=0.5)
        ax.plot(x, y, "k-", linewidth=1)

        ax.scatter(
            [z.real for z in vertices],
            [z.imag for z in vertices],
            color="black",
            s=20,
            zorder=5,
        )

        ax.axhline(0, color="k", linewidth=0.5)
        ax.axvline(0, color="k", linewidth=0.5)
        ax.grid(True, alpha=0.3)

        ax.set_xlim(-plotsize, plotsize)
        ax.set_ylim(-plotsize, plotsize)

        ax.set_title(f"Iteration {frame_num}", pad=12)

    anim = FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=duration,
        repeat=True,
    )

    plt.close()
    return anim.to_jshtml()

# =========================
# NEW Centered Animation
# =========================

def animate_folding_centered(
    mu,
    nu,
    iters,
    duration,
    plotsize=3,
    pointsize=2,
    orbit=False,
    iters_orbit=1000,
    alpha_orbit=0.3,
):
    if orbit:
        v0_orbit = -np.sqrt(1 + mu * nu - mu - nu) + 1j * mu
        v1_orbit = np.sqrt(1 + mu * nu - mu - nu) + 1j * nu
        v2_orbit = v1_orbit.conjugate()
        v3_orbit = v0_orbit.conjugate()

        center = (v0_orbit + v1_orbit + v2_orbit + v3_orbit) / 4
        v0_orbit -= center
        v1_orbit -= center
        v2_orbit -= center
        v3_orbit -= center

        orbit_points = [v0_orbit, v1_orbit, v2_orbit, v3_orbit]

        for _ in range(iters_orbit):
            v0_orbit, v1_orbit, v2_orbit, v3_orbit = fold_centered(
                v0_orbit, v1_orbit, v2_orbit, v3_orbit
            )
            orbit_points.extend([v0_orbit, v1_orbit, v2_orbit, v3_orbit])

        orbit_x = [z.real for z in orbit_points]
        orbit_y = [z.imag for z in orbit_points]

    v0 = -np.sqrt(1 + mu * nu - mu - nu) + 1j * mu
    v1 = np.sqrt(1 + mu * nu - mu - nu) + 1j * nu
    v2 = v1.conjugate()
    v3 = v0.conjugate()

    center = (v0 + v1 + v2 + v3) / 4
    v0 -= center
    v1 -= center
    v2 -= center
    v3 -= center

    frames = [(v0, v1, v2, v3)]

    for _ in range(iters):
        v0, v1, v2, v3 = fold_centered(v0, v1, v2, v3)
        frames.append((v0, v1, v2, v3))

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    fig.subplots_adjust(top=0.92)

    def update(frame_num):
        ax.clear()

        if orbit:
            ax.scatter(
                orbit_x,
                orbit_y,
                color="gray",
                s=pointsize,
                alpha=alpha_orbit,
            )

        v0, v1, v2, v3 = frames[frame_num]
        vertices = [v0, v1, v2, v3]

        x = [z.real for z in vertices] + [v0.real]
        y = [z.imag for z in vertices] + [v0.imag]

        ax.fill(x, y, color="lightgray", alpha=0.5)
        ax.plot(x, y, "k-", linewidth=1)

        ax.scatter(
            [z.real for z in vertices],
            [z.imag for z in vertices],
            color="black",
            s=20,
            zorder=5,
        )

        ax.axhline(0, color="k", linewidth=0.5)
        ax.axvline(0, color="k", linewidth=0.5)
        ax.grid(True, alpha=0.3)

        ax.set_xlim(-plotsize, plotsize)
        ax.set_ylim(-plotsize, plotsize)

        ax.set_title(f"Iteration {frame_num} (Centered)", pad=12)

    anim = FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=duration,
        repeat=True,
    )

    plt.close()
    return anim.to_jshtml()

# =========================
# Streamlit UI
# =========================

st.set_page_config(page_title="Iterated Folding Visualizer", layout="wide")

st.title("Iterated Folding Visualizer")

mode = st.radio(
    "",
    ["Plot Orbit", "Animate Folding", "Animate Folding (Centered)"],
    horizontal=True,
    label_visibility="collapsed",
)

col1, col2, col3 = st.columns(3)

with col1:
    mu = st.slider("μ (mu)", 0.0, 1.0, 0.3, 0.01)

with col2:
    nu = st.slider("ν (nu)", 0.0, 1.0, 0.4, 0.01)

with col3:
    plotsize = st.slider("Plot Size", 1, 20, 3, 1)

# =========================
# Mode 1: Plot Orbit
# =========================

if mode == "Plot Orbit":

    col1, col2 = st.columns(2)

    with col1:
        iters = st.slider("Iterations", 10, 5000, 2000, 10)

    with col2:
        pointsize = st.slider("Point Size", 1, 10, 5, 1)

    if st.button("Generate Orbit Plot", type="primary", use_container_width=True):
        fig = plot_orbit_to_image(mu, nu, iters, plotsize, pointsize)
        st.pyplot(fig)
        plt.close()

# =========================
# Mode 2: Animate Folding
# =========================

elif mode == "Animate Folding":

    col1, col2 = st.columns(2)

    with col1:
        iters = st.slider("Animation Iterations", 1, 100, 20, 1)

    with col2:
        duration = st.slider("Frame Duration (ms)", 50, 1000, 200, 50)

    orbit = st.checkbox("Show Orbit Background", value=False)

    if st.button("Generate Animation", type="primary", use_container_width=True):
        html_anim = animate_folding(mu, nu, iters, duration, plotsize, 2, orbit)
        st.components.v1.html(html_anim, height=900)

# =========================
# Mode 3: Animate Folding (Centered)
# =========================

elif mode == "Animate Folding (Centered)":

    col1, col2 = st.columns(2)

    with col1:
        iters = st.slider("Animation Iterations", 1, 100, 20, 1)

    with col2:
        duration = st.slider("Frame Duration (ms)", 50, 1000, 200, 50)

    orbit = st.checkbox("Show Orbit Background", value=False)

    if st.button("Generate Centered Animation", type="primary", use_container_width=True):
        html_anim = animate_folding_centered(
            mu, nu, iters, duration, plotsize, 2, orbit
        )
        st.components.v1.html(html_anim, height=900)
