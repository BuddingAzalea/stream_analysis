## Importing packages
import sys, os, errno, pickle, h5py, time, io, contextlib
from functools import wraps
from typing import Union

##analysis packages
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from scipy.ndimage import gaussian_filter1d

## custom packages. 
import agama

# define the physical units used in the code: the choice below corresponds to
# length scale = 1 kpc, velocity = 1 km/s, mass = 1 Msun
agama.setUnits(mass=1, length=1, velocity=1)

def generate_stream_coords(
    xv: np.ndarray,
    xv_prog: Union[np.ndarray, list] = [],
    degrees: bool = True,
    optimizer_fit: bool = False,
    fit_kwargs: dict | None = None,
) -> Union[
    tuple[np.ndarray, np.ndarray, float | None],
    tuple[np.ndarray, np.ndarray, np.ndarray | None]
]:
    """
    Vectorized version: Convert galactocentric phase space (x, y, z, vx, vy, vz)
    into stream-aligned coordinates (phi1, phi2) for single or multiple streams.

    Parameters
    ----------
    xv : np.ndarray, shape (N, 6) or (S, N, 6)
        Particle positions/velocities in galactocentric coordinates. S = number of streams.
    xv_prog : np.ndarray of shape (6,) or (S, 6), optional
        Progenitor phase space vector(s). If not provided, auto-estimated per stream.
    degrees : bool
        If True (default), angles are returned in degrees.
    optimizer_fit : bool
        If True, optimize rotation in phi1-phi2 plane per stream.
    fit_kwargs : dict
        Extra args for scipy.optimize.minimize

    Returns
    -------
    phi1 : np.ndarray
        Shape (N,) for single stream or (S, N) for multiple
    phi2 : np.ndarray
        Shape (N,) or (S, N)
    """
    xv = np.asarray(xv)
    is_batch = xv.ndim == 3  # (S, N, 6) or (N, 6)

    if not is_batch:
        # Single stream case: use existing logic
        return _generate_stream_coords_single(xv, xv_prog, degrees, optimizer_fit, fit_kwargs)

    # Multiple streams
    S, N, D = xv.shape
    assert D == 6, "Each particle must have 6 phase-space values"
    
    xv_prog = np.asarray(xv_prog)
    if xv_prog.size == 0:
        # Auto-detect progenitor per stream
        med = np.median(xv[:, :, :3], axis=1)  # (S, 3)
        dists = np.linalg.norm(xv[:, :, :3] - med[:, None, :], axis=2)  # (S, N)
        idxs = np.argmin(dists, axis=1)  # (S,)
        xv_prog = np.array([xv[s, idxs[s]] for s in range(S)])  # (S, 6)

    if xv_prog.ndim == 1:
        xv_prog = xv_prog[None, :]  # make (1,6) for broadcast

    assert xv_prog.shape == (S, 6), "xv_prog must be shape (S, 6) for batch mode"

    # Compute stream basis vectors for each progenitor
    L = np.cross(xv_prog[:, :3], xv_prog[:, 3:])  # (S, 3)
    L /= np.linalg.norm(L, axis=1)[:, None]       # Normalize (S, 3)

    xhat = xv_prog[:, :3] / np.linalg.norm(xv_prog[:, :3], axis=1)[:, None]  # (S, 3)
    zhat = L
    yhat = np.cross(zhat, xhat)  # (S, 3)

    # Stack into basis matrices: (S, 3, 3)
    R = np.stack([xhat, yhat, zhat], axis=-1)  # (S, 3, 3)

    # Project particles into new frame
    coords = np.einsum('sni,sij->snj', xv[:, :, :3], R)  # (S, N, 3)
    xs, ys, zs = coords[..., 0], coords[..., 1], coords[..., 2]
    rs = np.sqrt(xs**2 + ys**2 + zs**2)

    # Compute phi1, phi2
    phi1 = np.arctan2(ys, xs)
    phi2 = np.arcsin(zs / rs)

    # Optional rotation optimization
    theta_opt = None
    if optimizer_fit:
        from scipy.optimize import minimize
        theta_opt = np.empty(S)
        for s in range(S):
            def _cost_fn(theta):
                c, s_ = np.cos(theta), np.sin(theta)
                p1 =  c * phi1[s] - s_ * phi2[s]
                p2 =  s_ * phi1[s] + c * phi2[s]
                return np.sum(p2**2)

            res = minimize(_cost_fn, x0=0.0, **(fit_kwargs or {}))
            theta = res.x.item()
            theta_opt[s] = theta
            c, s_ = np.cos(theta), np.sin(theta)
            phi1[s], phi2[s] = c * phi1[s] - s_ * phi2[s], s_ * phi1[s] + c * phi2[s]

    # Convert to degrees if requested
    if degrees:
        phi1 = np.degrees(phi1)
        phi2 = np.degrees(phi2)
        if theta_opt is not None:
            theta_opt = np.degrees(theta_opt)

    return phi1, phi2

def _generate_stream_coords_single(
    xv: np.ndarray,
    xv_prog: Union[np.ndarray, list] = [],
    degrees: bool = True,
    optimizer_fit: bool = False,
    fit_kwargs: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, float | None]:
    """
    Convert galactocentric positions and velocities into stream-aligned coordinates (phi1, phi2),
    with an optional optimization to rotate the frame so as to minimize the variance in phi2.

    Parameters
    ----------
    xv : np.ndarray, shape (N, 6)
        Particle positions and velocities in galactocentric coordinates (x, y, z, vx, vy, vz).
    xv_prog : np.ndarray of shape (6,) or list, optional
        Progenitor’s position and velocity in galactocentric coordinates.
        If empty (default), the progenitor is taken as the particle
        whose 3D position is closest to the median of all xv[:,:3].
    degrees : bool, optional
        If True (default), angles phi1 and phi2 are returned in degrees; otherwise in radians.
    optimizer_fit : bool, optional
        If True, perform a 1D optimization over a rotation angle θ that minimizes the sum of φ₂².
    fit_kwargs : dict or None, optional
        Additional keyword args to pass to `scipy.optimize.minimize`, e.g.
        `{"method":"Powell", "options":{"maxiter":2000, "disp":True}}`.

    Returns
    -------
    phi1 : np.ndarray, shape (N,)
        Stream longitude (aligned with the progenitor’s orbit), after any fitted rotation.
    phi2 : np.ndarray, shape (N,)
        Stream latitude (perpendicular deviation from the main track), after any fitted rotation.
    theta_opt : float or None
        The best-fit rotation angle (in degrees) applied to the initial frame, or None if
        `optimizer_fit=False`.
    """

    # Determine progenitor if not provided
    if len(xv_prog) < 1:
        # pick the particle closest to the median position
        med = np.median(xv[:, :3], axis=0)
        idx = np.argmin(np.linalg.norm(xv[:, :3] - med, axis=1))
        xv_prog = xv[idx]

    # 1) Compute the “raw” stream frame basis from the progenitor
    L_prog = np.cross(xv_prog[:3], xv_prog[3:])
    L_prog /= np.linalg.norm(L_prog)

    xs_hat = xv_prog[:3] / np.linalg.norm(xv_prog[:3])
    zs_hat = L_prog
    ys_hat = np.cross(zs_hat, xs_hat)

    # 2) Project all particles into that frame
    coords = np.dot(xv[:, :3], np.vstack([xs_hat, ys_hat, zs_hat]).T)
    xs, ys, zs = coords[:,0], coords[:,1], coords[:,2]
    rs = np.sqrt(xs**2 + ys**2 + zs**2)

    # 3) Compute initial phi1, phi2
    phi1 = np.arctan2(ys, xs)
    phi2 = np.arcsin(zs / rs)

    # Optionally fit for a rotation θ around the φ1–φ2 plane
    theta_opt = None
    if optimizer_fit:
        from scipy.optimize import minimize
        # cost function: given θ (in radians), rotate (φ1, φ2) and sum φ2^2
        def _cost_fn(theta):
            c, s = np.cos(theta), np.sin(theta)
            # rotation matrix [[c, -s],[s, c]] acting on columns [φ1, φ2]
            p1 =  c*phi1 - s*phi2
            p2 =  s*phi1 + c*phi2
            return np.sum(p2**2)

        # initial guess θ=0, fit in radians
        fit_kwargs = fit_kwargs or {}
        res = minimize(
            _cost_fn, 
            x0=0.0,
            **fit_kwargs
        )
        theta_opt = res.x.item()  # in radians

        # apply the optimal rotation
        c, s = np.cos(theta_opt), np.sin(theta_opt)
        phi1, phi2 = (c*phi1 - s*phi2), (s*phi1 + c*phi2)

    # Convert to degrees if requested
    if degrees:
        phi1 = np.degrees(phi1)
        phi2 = np.degrees(phi2)
        if theta_opt is not None:
            theta_opt = np.degrees(theta_opt)

    return phi1, phi2 #, theta_opt

def get_observed_stream_coords(
    xv: np.ndarray,
    xv_prog: Union[np.ndarray, list] = [],
    degrees: bool = True,
    optimizer_fit: bool = False,
    fit_kwargs: dict | None = None,
    galcen_distance: float = 8.122,
    galcen_v_sun: tuple = (12.9, 245.6, 7.78),
    z_sun: float = 0.0208,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert galactocentric coords → observed (RA, Dec, v_los)
    + stream coords (phi1, phi2), fully vectorized via generate_stream_coords.
    """

    # Ensure a batch dimension
    is_batch = (xv.ndim == 3)
    if not is_batch:
        xv = xv[None, ...]  # (1, N, 6)

    S, N, _ = xv.shape

    # Handle progenitors in batch
    xv_prog = np.asarray(xv_prog)
    if xv_prog.size == 0:
        # auto-detect per-stream
        xv_prog = np.stack([
            stream[np.argmin(np.linalg.norm(stream[:, :3] - np.median(stream[:, :3], axis=0), axis=1))]
            for stream in xv
        ])
    elif xv_prog.ndim == 1:
        assert xv_prog.shape == (6,)
        xv_prog = np.broadcast_to(xv_prog, (S, 6))
    else:
        assert xv_prog.shape == (S, 6)

    # 1) Flatten for Agama transforms
    flat = xv.reshape(-1, 6)
    x, y, z, vx, vy, vz = flat.T

    # 2) Get Galactic → ICRS
    l, b, dist, pml, pmb, vlos = agama.getGalacticFromGalactocentric(
        x, y, z, vx, vy, vz,
        galcen_distance=galcen_distance,
        galcen_v_sun=galcen_v_sun,
        z_sun=z_sun,
    )
    ra, dec = agama.transformCelestialCoords(agama.fromGalactictoICRS, l, b)

    if degrees:
        ra = np.degrees(ra)
        dec = np.degrees(dec)

    # 3) Reshape back to (S, N)
    ra   = ra.reshape(S, N)
    dec  = dec.reshape(S, N)
    vlos = vlos.reshape(S, N)

    # 4) Compute (phi1, phi2) in one batch call
    phi1, phi2, theta_opt = generate_stream_coords(
        xv, xv_prog,
        degrees=degrees,
        optimizer_fit=optimizer_fit,
        fit_kwargs=fit_kwargs,
    )
    # phi1, phi2 are each shape (S, N) when xv.ndim==3

    # 5) Drop the batch axis if it was a single stream
    if not is_batch:
        return ra[0], dec[0], vlos[0], phi1[0], phi2[0]
    return ra, dec, vlos, phi1, phi2


def produce_stream_plot(xv_stream, color='ro', ax=None, xv_prog=[], 
                        alpha_lim=(None, None), delta_lim=(None, None),
                        Phi1_lim=(None, None), Phi2_lim=(None, None),
                        ms=0.5, mew=0,  
                       ):
    """
    Produce a 2×3 panel of stream diagnostic plots for a given particle set.

    Parameters
    ----------
    xv_stream : ndarray, shape (N, 6)
        Stream particle positions and velocities in galactocentric coordinates
        (x, y, z, vx, vy, vz).
    color : str, optional
        Matplotlib format string for line/marker color and style (default 'ro').
    ax : array_like of Axes or None, optional
        If provided, must be a (2,3) array of existing Matplotlib Axes to plot into.
        If None (default), a new figure and axes are created.
    xv_prog : array_like, shape (6,), optional
        Progenitor’s position and velocity for computing observed coordinates.
    alpha_lim : tuple of float or None, optional
        x-axis limits for RA plots in degrees (default (None, None)).
    delta_lim : tuple of float or None, optional
        y-axis limits for Dec plots in degrees (default (None, None)).
    Phi1_lim : tuple of float or None, optional
        x-axis limits for φ₁ stream longitude (default (None, None)).
    Phi2_lim : tuple of float or None, optional
        y-axis limits for φ₂ stream latitude (default (None, None)).
    ms : float, optional
        Marker size for scatter/line plots (default 0.5).
    mew : float, optional
        Marker edge width for scatter/line plots (default 0).

    Returns
    -------
    fig : matplotlib.figure.Figure or None
        Figure object if a new one was created, otherwise None.
    ax : ndarray of Axes, shape (2, 3)
        Array of Axes where the data have been plotted.
    """
    
    import matplotlib.pyplot as plt 
    if ax is None:
        fig, ax = plt.subplots(2, 3, figsize=(9, 6), dpi=300)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)        
        
    for i, ax_dummy in enumerate(ax.flatten()):
        if ax[i // ax.shape[1], i % ax.shape[1]] != ax[0, 1]:  # Skip ax[0, 1]
            ax_dummy.set_aspect('equal')
        
        ax[0, 0].set_xlabel(r'$\alpha$ [deg]')
        ax[0, 0].set_ylabel(r'$\delta$ [deg]')
        ax[0, 1].set_xlabel(r'$\alpha$ [deg]')
        ax[0, 1].set_ylabel(r'$v_{\rm los}$ [km/s]')
        ax[0, 2].set_xlabel(r'$\phi_1$ [deg]')
        ax[0, 2].set_ylabel(r'$\phi_2$ [deg]')

        ax[1, 0].set_xlabel('X [kpc]')
        ax[1, 0].set_ylabel('Y [kpc]')
        ax[1, 1].set_xlabel('X [kpc]')
        ax[1, 1].set_ylabel('Z [kpc]')
        ax[1, 2].set_xlabel('Y [kpc]')
        ax[1, 2].set_ylabel('Z [kpc]')

    ra, dec, vlos, phi1, phi2 = get_observed_stream_coords(xv_stream, xv_prog)
    
    
    ax[0, 0].plot(ra, dec, color, ms=ms, mew=mew)
    ax[0, 1].plot(ra, vlos, color, ms=ms, mew=mew)
    ax[0, 2].plot(phi1, phi2, color, ms=ms, mew=mew)

    ax[0, 0].set_xlim(alpha_lim)
    ax[0, 0].set_ylim(delta_lim)
    ax[0, 1].set_xlim(alpha_lim)
    
    ax[0, 2].set_xlim(Phi1_lim)
    ax[0, 2].set_ylim(Phi2_lim)
    
    ax[1, 0].plot(xv_stream[:,0], xv_stream[:,1], color, ms=ms, mew=mew)
    ax[1, 1].plot(xv_stream[:,0], xv_stream[:,2], color, ms=ms, mew=mew)
    ax[1, 2].plot(xv_stream[:,1], xv_stream[:,2], color, ms=ms, mew=mew)
    plt.tight_layout()

def return_stream_plots(Nbody_out, time_step=-1, x_axis=0, y_axis=2, LMC_traj=[], three_d_plot=False, interactive=False):

    """
    Generate a three‐panel evolution plot for an N‐body stream simulation.

    Parameters
    ----------
    Nbody_out : dict
        Dictionary containing simulation outputs with keys:
        - 'times' : array_like, shape (T,)
        - 'prog_xv' : ndarray, shape (T,6)
        - 'part_xv' : ndarray, shape (N,T,6) or (N,6)
        - 'bound_mass' : array_like, shape (T,), optional
    time_step : int, optional
        Index of the snapshot to plot for particle positions (default -1, last snapshot).
    x_axis : int, {0,1,2}, optional
        Coordinate index for the x‐axis in the right panel (0→X,1→Y,2→Z; default 0).
    y_axis : int, {0,1,2}, optional
        Coordinate index for the y‐axis in the right panel (default 2).
    LMC_traj : ndarray, shape (M,4) or empty, optional
        Trajectory of the LMC to overplot, where each row is (t, x, y, z).
        If empty (default), no LMC trajectory is drawn.
    three_d_plot : bool, optional
        If True, the middle panel is a 3D trajectory; otherwise it shows bound fraction vs time.
    interactive : bool, optional
        If True, enable interactive Matplotlib widget mode (requires IPython).
        Only valid when `three_d_plot=True`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    ax : list of Axes
        List of three Axes objects: [distance vs time, middle panel, right panel].
    """
    
    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks
    
    # Set up backend check
    if interactive and not three_d_plot:
        raise ValueError("Interactive mode only works with 3D plots")
        
    # Configure matplotlib backend
    if interactive:
        try:
            from IPython import get_ipython
            get_ipython().run_line_magic('matplotlib', 'widget')
        except ImportError:
            raise RuntimeError("Interactive mode requires IPython")
    
    # Create figure and axes
    fig = plt.figure(figsize=(12, 3), dpi=300)
    ax = [None, None, None]
    ax[0] = fig.add_subplot(131)
    ax[2] = fig.add_subplot(133)
    
    # Create middle axis with 3D projection if needed
    if three_d_plot:
        ax[1] = fig.add_subplot(132, projection='3d')
    else:
        ax[1] = fig.add_subplot(132)
    
    plt.subplots_adjust(wspace=0.25)

    # Common labels
    axes_label = {0:r'{\bf X [kpc]}', 1:r'{\bf Y [kpc]}', 2:r'{\bf Z [kpc]}'}
    ax[0].set_xlabel(r'{\bf T [Gyr]}'); ax[0].set_ylabel(r'{\bf $\mathbf{d_{cen}}$ [kpc]}')
    ax[2].set_xlabel(axes_label[x_axis]); ax[2].set_ylabel(axes_label[y_axis])

    # Configure middle axis labels
    if three_d_plot:
        ax[1].set_xlabel(axes_label[0])
        ax[1].set_ylabel(axes_label[1])
        ax[1].set_zlabel(axes_label[2])
        # ax[1].zaxis.set_label_position("left")
        
    else:
        ax[1].set_xlabel(r'{\bf T [Gyr]}'); ax[1].set_ylabel(r'{\bf Bound Frac}')

    # --- Plot common elements ---
    # Distance vs time plot
    distances = np.linalg.norm(Nbody_out['prog_xv'][:, :3], axis=1)
    ax[0].plot(Nbody_out['times'][:], distances, c='r', lw=0.5)
    
    # --- Middle panel content ---
    if not three_d_plot:
        try:  # Original bound fraction plot
            bound_frac = Nbody_out['bound_mass'][:] / Nbody_out['bound_mass'][0]
            ax[1].plot(Nbody_out['times'][:], bound_frac, c='r', lw=0.5)
        except Exception as e:
            ax[1].plot(Nbody_out['times'][:], np.zeros_like(Nbody_out['times'][:]), c='r', lw=0.5)
    else:  # 3D trajectory plot
        from mpl_toolkits.mplot3d import Axes3D  # Required for 3D projection
        # Plot progenitor trajectory
        prog_x = Nbody_out['prog_xv'][:, 0]
        prog_y = Nbody_out['prog_xv'][:, 1]
        prog_z = Nbody_out['prog_xv'][:, 2]
        
        if time_step == -1:
            ax[1].plot(prog_x, prog_y, prog_z, c='r', lw=0.5)
        else:
            ax[1].plot(prog_x[:time_step+1], prog_y[:time_step+1], prog_z[:time_step+1], c='r', lw=0.5)
        
        # Plot particles
        try:
            part_x = Nbody_out['part_xv'][:, time_step, 0]
            part_y = Nbody_out['part_xv'][:, time_step, 1]
            part_z = Nbody_out['part_xv'][:, time_step, 2]
        except Exception as e:
            part_x = Nbody_out['part_xv'][:, 0]
            part_y = Nbody_out['part_xv'][:, 1]
            part_z = Nbody_out['part_xv'][:, 2]
        ax[1].scatter(part_x, part_y, part_z, s=0.25, c='m')
        # Make axis panes transparent
        ax[1].xaxis.pane.fill = False
        ax[1].yaxis.pane.fill = False
        ax[1].zaxis.pane.fill = False
        ax[1].grid(False) # This line turns off the grid
        
        # Automatically determine best view angle using PCA
        try:
            # Get particle coordinates
            try:
                part_x = Nbody_out['part_xv'][:, time_step, 0]
                part_y = Nbody_out['part_xv'][:, time_step, 1]
                part_z = Nbody_out['part_xv'][:, time_step, 2]
            except Exception as e:
                part_x = Nbody_out['part_xv'][:, 0]
                part_y = Nbody_out['part_xv'][:, 1]
                part_z = Nbody_out['part_xv'][:, 2]

            # Compute PCA
            data = np.vstack([part_x, part_y, part_z]).T
            data_centered = data - np.mean(data, axis=0)
            
            if len(data) > 1 and not np.allclose(data_centered, 0):
                cov = np.cov(data_centered, rowvar=False)
                eigen_vals, eigen_vecs = np.linalg.eigh(cov)
                order = eigen_vals.argsort()[::-1]
                eigen_vecs = eigen_vecs[:, order]
                
                # Get third principal component direction
                pc3 = eigen_vecs[:, 2]
                direction = -pc3  # View along PC3 direction
                dx, dy, dz = direction

                # Calculate angles
                azim = np.degrees(np.arctan2(dy, dx))
                r = np.linalg.norm(direction)
                phi = np.arccos(dz/r) if r != 0 else 0
                elev = 90 - np.degrees(phi)

                # Apply view angle
                ax[1].view_init(elev=max(0, min(90, elev)), 
                            azim=azim % 360)
            else:
                ax[1].view_init(elev=30, azim=45)
        except Exception as e:
            ax[1].view_init(elev=30, azim=45)
        

    # --- Common pericenter calculations ---
    inverted_distances = -distances
    peaks, _ = find_peaks(inverted_distances)
    if len(peaks) > 0:
        for peak in peaks:
            ax[0].axvline(Nbody_out['times'][peak], color='gray', lw=0.5, alpha=0.5, ls='--')
            if not three_d_plot:
                ax[1].axvline(Nbody_out['times'][peak], color='gray', lw=0.5, alpha=0.5, ls='--')

    # --- Right panel content ---
    # Initial conditions star
    ax[2].scatter(Nbody_out['prog_xv'][0, x_axis], Nbody_out['prog_xv'][0, y_axis], 
                  facecolor='none', edgecolor='r', s=50, marker='*', linewidth=0.3, zorder=3)
    ax[2].scatter(Nbody_out['prog_xv'][-1, x_axis], Nbody_out['prog_xv'][-1, y_axis], 
                  facecolor='none', edgecolor='k', s=50, marker='*', linewidth=0.3, zorder=3)
    
    # Particles scatter
    try:
        ax[2].scatter(Nbody_out['part_xv'][:, time_step, x_axis], 
                      Nbody_out['part_xv'][:, time_step, y_axis], s=0.25, c='m')
    except Exception as e:
        ax[2].scatter(Nbody_out['part_xv'][:, x_axis], 
                      Nbody_out['part_xv'][:, y_axis], s=0.25, c='m')
    
    # Progenitor trajectory
    if time_step == -1:
        ax[2].plot(Nbody_out['prog_xv'][:, x_axis], Nbody_out['prog_xv'][:, y_axis], lw=0.5, ls='--', c='gray')
    else:
        ax[2].plot(Nbody_out['prog_xv'][:time_step+1, x_axis], 
                   Nbody_out['prog_xv'][:time_step+1, y_axis], lw=0.5, ls='--', c='gray')

    # --- Axis limit management ---
    # For 3D plot
    if three_d_plot:
        ax[1].relim()
        ax[1].autoscale_view()
        ax[1].set_autoscale_on(False)  # Lock 3D axis limits
    
    # For right panel (2D)
    ax[2].relim()
    ax[2].autoscale_view()
    ax[2].autoscale(False)
    
    # For left panel
    ax[0].autoscale(False)

    # --- LMC trajectory handling ---
    if len(LMC_traj) > 0:
        # Left panel
        ax[0].plot(LMC_traj[:, 0], np.linalg.norm(LMC_traj[:, 1:], axis=1), 
                   c='gray', lw=2, alpha=0.4, ls='--')
        
        # Right panel
        ax[2].plot(LMC_traj[:, x_axis+1], LMC_traj[:, y_axis+1], 
                   c='gray', lw=2, alpha=0.4, ls='--')
        ax[2].scatter(LMC_traj[-1, x_axis+1], LMC_traj[-1, y_axis+1], 
                      facecolor='none', edgecolor='gray', s=50, marker='*', linewidth=1, zorder=3)
        
        # 3D panel
        if three_d_plot:
            ax[1].plot(LMC_traj[:, 1], LMC_traj[:, 2], LMC_traj[:, 3], 
                       c='gray', lw=2, alpha=0.4, ls='--')
            ax[1].scatter(LMC_traj[-1, 1], LMC_traj[-1, 2], LMC_traj[-1, 3], 
                          facecolor='none', edgecolor='gray', s=50, marker='*', linewidth=1, zorder=3)

    # Green reference lines
    if x_axis == 2 or y_axis == 2:
        ax[2].plot([-10, 10], [0, 0], c='g', alpha=0.5)

        
    # Interactive controls setup
    if interactive and three_d_plot:
        # Enable mouse interaction
        ax[1].mouse_init()
        
        # Optional: Add rotation controls
        def on_move(event):
            if event.inaxes == ax[1]:
                ax[1].view_init(elev=ax[1].elev, azim=ax[1].azim)
                
        fig.canvas.mpl_connect('motion_notify_event', on_move)
     
    # Don't block execution in interactive mode
    if not interactive:
        plt.show()
    else:
        plt.show(block=False)           


def measure_stream_LengthWidth(
    phi1: np.ndarray,
    phi2: np.ndarray,
    method: str = "quantile",         # "quantile" or "kde"
    density_frac: float = 0.95,
    nbins: int = 100,                 # number of bins per stream (KDE only)
    smoothing_sigma: float = 1.0,     # Gaussian σ in bin‐units (KDE only)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Measure stream length (along phi1) and width (along phi2), for one or many streams.

    You can choose:
    - "quantile": length and width are simple [(1−f)/2, (1+f)/2] quantiles.
    - "kde": length is the span of the phi1 region enclosing `density_frac`
             using a per-stream smoothed histogram (approximate KDE),
             and width is the quantile width of phi2 within that region.

    Parameters
    ----------
    phi1 : np.ndarray, shape (N,) or (S, N)
        Along-stream coordinates (degrees or radians).
    phi2 : np.ndarray, shape (N,) or (S, N)
        Across-stream coordinates.
    method : {"quantile","kde"}
        Which estimator to use.
    density_frac : float
        Fraction of total particles to enclose (e.g. 0.95).
    nbins : int
        Number of bins for the per-stream histogram (KDE mode).
    smoothing_sigma : float
        σ of the Gaussian filter, in bin units (KDE mode).

    Returns
    -------
    lengths : np.ndarray of shape (S,) or scalar
        Enclosed span in phi1.
    widths  : np.ndarray of shape (S,) or scalar
        Span in phi2, measured as quantile width within the KDE-defined phi1 region.
    """
    single = (phi1.ndim == 1)
    phi1_arr = np.atleast_2d(phi1)
    phi2_arr = np.atleast_2d(phi2)
    S, N = phi1_arr.shape
    assert phi2_arr.shape == (S, N), "phi1/phi2 must match shapes"

    if method == "quantile":
        # -------------------------
        # 1) Compute quantile-based length & width
        # -------------------------
        lo = (1 - density_frac) / 2
        hi = 1 - lo
        q1 = np.quantile(phi1_arr, [lo, hi], axis=1)  # phi1 quantiles
        q2 = np.quantile(phi2_arr, [lo, hi], axis=1)  # phi2 quantiles
        lengths = q1[1] - q1[0]
        widths  = q2[1] - q2[0]

    elif method == "kde":
        # -------------------------
        # 1) Determine per-stream min, max, and bin width for phi1
        # -------------------------
        min1 = phi1_arr.min(axis=1)
        max1 = phi1_arr.max(axis=1)
        w1   = (max1 - min1) / nbins  # per-stream bin width

        # -------------------------
        # 2) Digitize phi1 values into bin indices [0, nbins-1]
        # -------------------------
        rel = (phi1_arr - min1[:, None]) / w1[:, None]
        inds = np.floor(rel).astype(int)
        inds = np.clip(inds, 0, nbins - 1)

        # -------------------------
        # 3) Build per-stream histograms via a single bincount
        # -------------------------
        flat = inds + np.arange(S)[:, None] * nbins
        hist = np.bincount(flat.ravel(), minlength=S * nbins).reshape(S, nbins)

        # -------------------------
        # 4) Smooth histogram = approximate KDE
        # -------------------------
        smooth = gaussian_filter1d(hist.astype(float), smoothing_sigma, axis=1)

        # -------------------------
        # 5) Normalize to get a probability density over phi1
        # -------------------------
        area = smooth.sum(axis=1) * w1
        pdf  = smooth / area[:, None]

        # -------------------------
        # 6) Find threshold that encloses density_frac
        # -------------------------
        #   sort bins by density descending
        order = np.argsort(pdf, axis=1)[:, ::-1]
        #   cumulative density in sorted order
        cum   = np.cumsum(np.take_along_axis(pdf, order, axis=1) * w1[:, None], axis=1)
        #   index where cumulative ≥ density_frac
        idx   = np.argmax(cum >= density_frac, axis=1)
        #   threshold density per stream
        thr   = pdf[np.arange(S), order[np.arange(S), idx]]

        # -------------------------
        # 7) Mask bins above threshold to define the coherent phi1 region
        # -------------------------
        mask  = pdf >= thr[:, None]
        #   first and last True per stream
        first = mask.argmax(axis=1)
        last  = nbins - 1 - mask[:, ::-1].argmax(axis=1)

        # -------------------------
        # 8) Compute phi1 length: (last − first) × per-stream bin width
        # -------------------------
        lengths = (last - first) * w1

        # -------------------------
        # 9) Compute phi2 width by quantiles within the phi1 region
        # -------------------------
        lo = (1 - density_frac) / 2
        hi = 1 - lo
        widths = np.empty(S, dtype=float)
        for i in range(S):
            # select particles within the KDE-defined phi1 span
            mask_particles = (
                (phi1_arr[i] >= min1[i] + first[i] * w1[i]) &
                (phi1_arr[i] <= min1[i] + last[i]  * w1[i])
            )
            if mask_particles.sum() < 2:
                widths[i] = 0.0
            else:
                q2 = np.quantile(phi2_arr[i, mask_particles], [lo, hi])
                widths[i] = float(q2[1] - q2[0])
    else:
        raise ValueError("method must be 'quantile' or 'kde'")

    # -------------------------
    # 10) Return scalars if input was 1D
    # -------------------------
    if single:
        return float(lengths[0]), float(widths[0])
    return lengths, widths
