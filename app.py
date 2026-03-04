import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import integrate

st.set_page_config(page_title="Iterated Folding Visualizer", layout="wide")

st.title("Cyclic Folding")


# =========================
# Folding Functions
# =========================

def fold(v0, v1, v2, v3):
    v0_folded = (np.conj(v0 - v1) * (v3 - v1)) / np.conj(v3 - v1) + v1
    new_v0 = v1
    new_v1 = v2
    new_v2 = v3
    new_v3 = v0_folded
    return new_v0, new_v1, new_v2, new_v3


def fold_centered(v0, v1, v2, v3):
    v0, v1, v2, v3 = fold(v0, v1, v2, v3)
    center = (v0 + v1 + v2 + v3) / 4
    return v0 - center, v1 - center, v2 - center, v3 - center


# =========================
# Orbit Plot
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
# Animation (Standard)
# =========================

def animate_folding(
    mu, nu, iters, duration, plotsize=3,
    pointsize=2, orbit=False,
    iters_orbit=1000, alpha_orbit=0.3,
):
    if orbit:
        v0_o = -np.sqrt(1 + mu * nu - mu - nu) + 1j * mu
        v1_o = np.sqrt(1 + mu * nu - mu - nu) + 1j * nu
        v2_o = v1_o.conjugate()
        v3_o = v0_o.conjugate()

        orbit_points = [v0_o, v1_o, v2_o, v3_o]

        for _ in range(iters_orbit):
            v0_o, v1_o, v2_o, v3_o = fold(v0_o, v1_o, v2_o, v3_o)
            orbit_points.extend([v0_o, v1_o, v2_o, v3_o])

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
            ax.scatter(orbit_x, orbit_y, color="gray", s=pointsize, alpha=alpha_orbit)

        v0, v1, v2, v3 = frames[frame_num]
        vertices = [v0, v1, v2, v3]

        x = [z.real for z in vertices] + [v0.real]
        y = [z.imag for z in vertices] + [v0.imag]

        ax.fill(x, y, color="lightgray", alpha=0.5)
        ax.plot(x, y, "k-", linewidth=1)
        ax.scatter([z.real for z in vertices], [z.imag for z in vertices],
                   color="black", s=20, zorder=5)

        ax.axhline(0, color="k", linewidth=0.5)
        ax.axvline(0, color="k", linewidth=0.5)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-plotsize, plotsize)
        ax.set_ylim(-plotsize, plotsize)
        ax.set_title(f"Iteration {frame_num}", pad=12)

    anim = FuncAnimation(fig, update, frames=len(frames), interval=duration, repeat=True)
    plt.close()
    return anim.to_jshtml()


# =========================
# Animation (Centered)
# =========================

def animate_folding_centered(
    mu, nu, iters, duration, plotsize=2.0,
    pointsize=2, orbit=False,
    iters_orbit=1000, alpha_orbit=0.3,
):
    if orbit:
        v0_o = -np.sqrt(1 + mu * nu - mu - nu) + 1j * mu
        v1_o = np.sqrt(1 + mu * nu - mu - nu) + 1j * nu
        v2_o = v1_o.conjugate()
        v3_o = v0_o.conjugate()

        center = (v0_o + v1_o + v2_o + v3_o) / 4
        v0_o -= center
        v1_o -= center
        v2_o -= center
        v3_o -= center

        orbit_points = [v0_o, v1_o, v2_o, v3_o]

        for _ in range(iters_orbit):
            v0_o, v1_o, v2_o, v3_o = fold_centered(v0_o, v1_o, v2_o, v3_o)
            orbit_points.extend([v0_o, v1_o, v2_o, v3_o])

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
            ax.scatter(orbit_x, orbit_y, color="gray", s=pointsize, alpha=alpha_orbit)

        v0, v1, v2, v3 = frames[frame_num]
        vertices = [v0, v1, v2, v3]

        x = [z.real for z in vertices] + [v0.real]
        y = [z.imag for z in vertices] + [v0.imag]

        ax.fill(x, y, color="lightgray", alpha=0.5)
        ax.plot(x, y, "k-", linewidth=1)
        ax.scatter([z.real for z in vertices], [z.imag for z in vertices],
                   color="black", s=20, zorder=5)

        ax.axhline(0, color="k", linewidth=0.5)
        ax.axvline(0, color="k", linewidth=0.5)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-plotsize, plotsize)
        ax.set_ylim(-plotsize, plotsize)
        ax.set_title(f"Iteration {frame_num} (Centered)", pad=12)

    anim = FuncAnimation(fig, update, frames=len(frames), interval=duration, repeat=True)
    plt.close()
    return anim.to_jshtml()


# =========================
# Diagonal Dynamics Helpers
# =========================

def _dd_initial_vertices(mu, nu):
    v0 = -np.sqrt(1 + mu * nu - mu - nu) + 1j * mu
    v1 =  np.sqrt(1 + mu * nu - mu - nu) + 1j * nu
    v2 = v1.conjugate()
    v3 = v0.conjugate()
    center = (v0 + v1 + v2 + v3) / 4
    return v0 - center, v1 - center, v2 - center, v3 - center


def _dd_diagonal_pair(v0, v1, v2, v3):
    x = abs(v0 - v2)**2
    y = abs(v1 - v3)**2
    return x, y


def _dd_QLC(mu, nu):
    s = 2 - mu - nu
    Q = 4*(mu**2 + nu**2) + 2*s**2
    L = (4*mu**2 - s**2) * (4*nu**2 - s**2)
    C = (16*mu**2*nu**2 - s**4) * (4*(mu**2 + nu**2) - 2*s**2)
    return Q, L, C


def _dd_g2g3(mu, nu):
    Q, L, C = _dd_QLC(mu, nu)
    g2 = (1/12)*Q**4 + (4/3)*Q**2*L + (4/3)*L**2 - 2*Q*C
    g3 = ((-1/216)*Q**6 - (1/9)*Q**4*L + (1/6)*Q**3*C
          - (5/9)*L**2*Q**2 + (8/27)*L**3 + (4/3)*C*L*Q - C**2)
    return g2, g3


def _dd_Hx(mu, nu):
    Q, L, C = _dd_QLC(mu, nu)
    return (Q**2 + 8*L) / 12


def _dd_integrand(t, g2, g3):
    val = 4*t**3 - g2*t - g3
    return 1.0 / np.sqrt(val) if val > 0 else 0.0


def _dd_compute_rho(mu, nu):
    Q, L, C = _dd_QLC(mu, nu)
    g2, g3  = _dd_g2g3(mu, nu)
    hx      = _dd_Hx(mu, nu)

    roots = np.roots([4, 0, -g2, -g3])
    real_roots = roots[np.abs(roots.imag) < 1e-6].real
    if len(real_roots) == 0:
        return np.nan
    e2 = np.max(real_roots)

    num, _ = integrate.quad(_dd_integrand, hx, np.inf, args=(g2, g3))
    den, _ = integrate.quad(_dd_integrand, e2, np.inf, args=(g2, g3))

    raw = num / (2 * den)
    return 1.0 - raw if C < 0 else raw


def _dd_is_degenerate(mu, nu, tol=1e-9):
    return abs(mu + nu - 1) < tol or abs(mu - nu) < tol


def diagonal_dynamics_animation(mu, nu, iters, duration_ms, quad_window=1.5, resolution=800, fade_steps=8):
    degenerate = _dd_is_degenerate(mu, nu)

    if not degenerate:
        rho = _dd_compute_rho(mu, nu)
        theta_step = 2 * np.pi * rho

    v0, v1, v2, v3 = _dd_initial_vertices(mu, nu)
    frames = [(v0, v1, v2, v3)]
    orbit_x, orbit_y = [], []

    for i in range(iters + 1):
        if i > 0:
            v0, v1, v2, v3 = fold_centered(v0, v1, v2, v3)
            frames.append((v0, v1, v2, v3))

        x_diag, y_diag = _dd_diagonal_pair(v0, v1, v2, v3)
        if i % 2 == 1:
            x_diag, y_diag = y_diag, x_diag

        orbit_x.append(x_diag)
        orbit_y.append(y_diag)

    orbit_x = np.array(orbit_x)
    orbit_y = np.array(orbit_y)

    curve_pad = 1.0
    Q, L, C = _dd_QLC(mu, nu)

    def F(x, y):
        return x**2*y + x*y**2 - Q*x*y - L*(x+y) + C

    xmin_raw = orbit_x.min() - curve_pad
    xmax_raw = orbit_x.max() + curve_pad
    ymin_raw = orbit_y.min() - curve_pad
    ymax_raw = orbit_y.max() + curve_pad

    x_width = xmax_raw - xmin_raw
    y_width = ymax_raw - ymin_raw

    if x_width < y_width:
        diff = (y_width - x_width) / 2
        xmin, xmax = xmin_raw - diff, xmax_raw + diff
        ymin, ymax = ymin_raw, ymax_raw
    else:
        diff = (x_width - y_width) / 2
        xmin, xmax = xmin_raw, xmax_raw
        ymin, ymax = ymin_raw - diff, ymax_raw + diff

    x_vals = np.linspace(xmin, xmax, resolution)
    y_vals = np.linspace(ymin, ymax, resolution)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = F(X, Y)

    if not degenerate:
        circle_pts = []
        theta = 0
        for i in range(iters + 1):
            circle_pts.append((np.cos(theta), np.sin(theta)))
            if i % 2 == 0 and i > 0:
                theta += theta_step
        circle_pts = np.array(circle_pts)

    n_panels = 2 if degenerate else 3
    # Smaller figsize so the whole animation fits without being cut off
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    fig.tight_layout(pad=2.0)

    ax_quad  = axes[0]
    ax_curve = axes[1]
    ax_circle = axes[2] if not degenerate else None

    def draw_fading_path(ax, xs, ys, i):
        for k in range(i):
            alpha = (0.5 + 0.5*(k - (i - fade_steps)) / fade_steps
                     if k >= i - fade_steps else 0.5)
            ax.plot(xs[k:k+2], ys[k:k+2],
                    color="gray", linewidth=1.4, alpha=alpha,
                    solid_capstyle='round', solid_joinstyle='round', zorder=1)

    def update(i):
        ax_quad.clear()
        v0, v1, v2, v3 = frames[i]
        verts = [v0, v1, v2, v3]
        xq = [z.real for z in verts] + [v0.real]
        yq = [z.imag for z in verts] + [v0.imag]

        ax_quad.fill(xq, yq, color="lightgray", alpha=0.5)
        ax_quad.plot(xq, yq, "k-", linewidth=1)
        ax_quad.plot([v0.real, v2.real], [v0.imag, v2.imag], "k:", linewidth=1)
        ax_quad.plot([v1.real, v3.real], [v1.imag, v3.imag], "k:", linewidth=1)
        ax_quad.scatter([z.real for z in verts], [z.imag for z in verts],
                        color="black", s=30, zorder=5)
        ax_quad.set_xlim(-quad_window, quad_window)
        ax_quad.set_ylim(-quad_window, quad_window)
        ax_quad.set_aspect("equal")
        ax_quad.grid(True, alpha=0.3)
        ax_quad.set_title(f"Cyclic Folding\nIteration {i}")

        ax_curve.clear()
        draw_fading_path(ax_curve, orbit_x, orbit_y, i)
        ax_curve.contour(X, Y, Z, levels=[0], colors="black", linewidths=1.5, zorder=3)
        ax_curve.scatter(orbit_x[i], orbit_y[i], color="black", s=40, zorder=4)
        ax_curve.set_xlim(xmin, xmax)
        ax_curve.set_ylim(ymin, ymax)
        ax_curve.set_aspect("equal")
        ax_curve.grid(True, alpha=0.3)
        ax_curve.set_title("Dynamics of π on Σ")

        if not degenerate:
            ax_circle.clear()
            draw_fading_path(ax_circle, circle_pts[:, 0], circle_pts[:, 1], i)
            circle_patch = plt.Circle((0, 0), 1, fill=False, color="black",
                                      linewidth=1.5, zorder=3)
            ax_circle.add_patch(circle_patch)
            ax_circle.scatter(circle_pts[i, 0], circle_pts[i, 1],
                               color="black", s=40, zorder=4)
            ax_circle.set_xlim(-1.2, 1.2)
            ax_circle.set_ylim(-1.2, 1.2)
            ax_circle.set_aspect("equal")
            ax_circle.grid(True, alpha=0.3)
            ax_circle.set_title(f"Rotation on Circle csp. τ\nρ ≈ {rho:.5f}")

    anim = FuncAnimation(fig, update, frames=iters + 1,
                         interval=duration_ms, repeat=True)
    plt.close()
    return anim.to_jshtml(), degenerate


# =========================
# Streamlit UI
# =========================

mode = st.radio(
    "",
    ["Plot Orbit", "Animate Folding", "Animate Folding (Centered)", "Visualize Diagonal Dynamics"],
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
# Plot Orbit Mode
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
# Animation Modes
# =========================

elif mode in ("Animate Folding", "Animate Folding (Centered)"):

    col1, col2, col3 = st.columns(3)

    with col1:
        iters = st.slider("Animation Iterations", 1, 100, 20, 1)

    with col2:
        duration = st.slider("Frame Duration (ms)", 50, 1000, 200, 50)

    with col3:
        if mode == "Animate Folding (Centered)":
            plotsize = st.slider("Quadrilateral Plot Size", 1.0, 3.0, 2.0, 0.25)
        else:
            pointsize = st.slider("Point Size", 1, 10, 2, 1)

    col1, col2, col3 = st.columns(3)

    with col1:
        orbit = st.checkbox("Show Orbit Background", value=False)

    with col2:
        iters_orbit = (
            st.slider("Orbit Iterations", 100, 5000, 2000, 100)
            if orbit else 2000
        )

    with col3:
        alpha_orbit = (
            st.slider("Orbit Transparency", 0.0, 1.0, 0.3, 0.05)
            if orbit else 0.3
        )

    if mode == "Animate Folding":
        label = "Generate Animation"
        func = animate_folding
    else:
        label = "Generate Centered Animation"
        func = animate_folding_centered

    if st.button(label, type="primary", use_container_width=True):
        html_anim = func(
            mu, nu, iters, duration,
            plotsize, pointsize if mode == "Animate Folding" else 2,
            orbit, iters_orbit, alpha_orbit
        )
        st.components.v1.html(html_anim, height=600)


# =========================
# Visualize Diagonal Dynamics
# =========================

else:

    col1, col2, col3 = st.columns(3)

    with col1:
        mu = st.slider("μ (mu)", 0.0, 1.0, 0.3, 0.01)
    
    with col2:
        nu = st.slider("ν (nu)", 0.0, 1.0, 0.4, 0.01)
    
    with col3:
        if mode != "Visualize Diagonal Dynamics":
            plotsize = st.slider("Plot Size", 1, 20, 3, 1)
        else:
            plotsize = 3  # unused default

    if _dd_is_degenerate(mu, nu):
        st.warning(
            "Degenerate case detected (μ + ν = 1 or μ = ν). "
            "The rotation number is undefined and the circle panel will be hidden."
        )

    if st.button("Generate Diagonal Dynamics", type="primary", use_container_width=True):
        html_anim, degen = diagonal_dynamics_animation(mu, nu, iters, duration, quad_window)
        st.components.v1.html(html_anim, height=600)
