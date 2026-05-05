"""
trajectory_viewer.py
Captures Langevin trajectory frames from a LangevinMol and exports an animated
Plotly HTML visualization of the binding event, plus a VRMSD convergence curve.

VRMSD (Voxel RMSD): the MD-equivalent of RMSD convergence. Instead of measuring
atom-to-atom Cartesian distance from a reference pose, we measure the Frobenius
distance between the current ligand demand grid and the minimum-energy (best) pose
grid. When VRMSD plateaus near zero, the molecule has converged into the binding mode.

    VRMSD(t) = sqrt( mean( (phi_ligand(t) - phi_best)^2 ) )

This is the voxel analog of RMSD convergence in MD trajectory analysis.
"""
import ctypes
import time
import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class TrajectoryCapture:
    """
    Captures Langevin trajectory frames by running nibble_langevin_step
    step-by-step from Python, recording position and score at each frame.

    Usage:
        cap = TrajectoryCapture(protein_engine, langevin_mol, n_steps=500, record_every=10)
        result = cap.run()
        cap.export_html("docking_trajectory.html", protein_atoms=..., start_coords=...)
    """
    def __init__(self, protein_engine, langevin_mol, n_steps=500, record_every=5,
                 dt=0.002, gamma=1.0, kT=0.593):
        self.engine = protein_engine
        self.mol = langevin_mol
        self.n_steps = n_steps
        self.record_every = record_every
        self.dt = dt
        self.gamma = gamma
        self.kT = kT

        self.frames_cx = []
        self.frames_cy = []
        self.frames_cz = []
        self.frames_score = []
        self.vrmsd = []

        self._lib = protein_engine._lib
        self._rng = ctypes.c_uint(12345)

        # Setup nibble_langevin_step binding if not already present
        from key.nibble_bridge import NibbleGridStruct, NibbleMolStruct
        _G = ctypes.POINTER(NibbleGridStruct)
        _M = ctypes.POINTER(NibbleMolStruct)
        _F = ctypes.c_float
        _PU = ctypes.POINTER(ctypes.c_uint)
        try:
            self._lib.nibble_langevin_step.argtypes = [_M, _G, _G, _F, _F, _F, _PU]
            self._lib.nibble_langevin_step.restype = None
        except Exception:
            pass

        # Scratch grid for scoring
        self._scratch = self._lib.nibble_create_grid(
            protein_engine.dim_x, protein_engine.dim_y,
            protein_engine.dim_z, protein_engine.resolution
        )

    def __del__(self):
        if hasattr(self, '_scratch') and self._scratch:
            self._lib.nibble_free_grid(self._scratch)

    def _score_current(self):
        from key.nibble_bridge import NibbleMolStruct
        return self._lib.nibble_mol_score(
            self.mol.mol_ptr, self.engine.grid_ptr, self._scratch
        )

    def run(self):
        """Run the trajectory and capture frames. Returns best_score, best_pos tuple."""
        best_score = self._score_current()
        best_cx = float(self.mol.mol_ptr.contents.cx)
        best_cy = float(self.mol.mol_ptr.contents.cy)
        best_cz = float(self.mol.mol_ptr.contents.cz)
        best_grid_snapshot = None

        # Record starting frame
        self.frames_cx.append(best_cx)
        self.frames_cy.append(best_cy)
        self.frames_cz.append(best_cz)
        self.frames_score.append(float(best_score))
        self.vrmsd.append(0.0)

        from key.nibble_bridge import N_CHANNELS
        n_vox = (self.engine.dim_x * self.engine.dim_y *
                 self.engine.dim_z * N_CHANNELS)

        for step in range(self.n_steps):
            self._lib.nibble_langevin_step(
                self.mol.mol_ptr, self.engine.grid_ptr, self._scratch,
                ctypes.c_float(self.dt), ctypes.c_float(self.gamma),
                ctypes.c_float(self.kT), ctypes.byref(self._rng)
            )

            sc = self._score_current()

            if sc < best_score:
                best_score = sc
                best_cx = float(self.mol.mol_ptr.contents.cx)
                best_cy = float(self.mol.mol_ptr.contents.cy)
                best_cz = float(self.mol.mol_ptr.contents.cz)
                # Snapshot the scratch grid at best pose
                best_grid_snapshot = np.ctypeslib.as_array(
                    self._scratch.contents.data, shape=(n_vox,)
                ).copy()

            if step % self.record_every == 0:
                cx = float(self.mol.mol_ptr.contents.cx)
                cy = float(self.mol.mol_ptr.contents.cy)
                cz = float(self.mol.mol_ptr.contents.cz)
                self.frames_cx.append(cx)
                self.frames_cy.append(cy)
                self.frames_cz.append(cz)
                self.frames_score.append(float(sc))

                # VRMSD relative to best pose grid
                if best_grid_snapshot is not None:
                    current_arr = np.ctypeslib.as_array(
                        self._scratch.contents.data, shape=(n_vox,)
                    )
                    diff = current_arr - best_grid_snapshot
                    vr = float(np.sqrt(np.mean(diff * diff)))
                else:
                    vr = 0.0
                self.vrmsd.append(vr)

        self._best_score = best_score
        self._best_pos = (best_cx, best_cy, best_cz)
        return best_score, (best_cx, best_cy, best_cz)

    def export_html(self, filename, protein_atoms=None, start_coords=(0.0, 0.0, 0.0),
                    resolution=0.25):
        """
        Exports an animated Plotly HTML with:
          - Animated scatter trace of the drug CoM path through the pocket
          - Score vs frame curve
          - VRMSD convergence curve
          - Optional protein atom cloud overlay
        """
        if not PLOTLY_AVAILABLE:
            print("plotly required: pip install plotly")
            return

        cx_arr = np.array(self.frames_cx)
        cy_arr = np.array(self.frames_cy)
        cz_arr = np.array(self.frames_cz)
        sc_arr = np.array(self.frames_score)
        vr_arr = np.array(self.vrmsd)
        n_frames = len(cx_arr)

        # Convert grid-space coords to world-space
        wx = cx_arr * resolution + start_coords[0]
        wy = cy_arr * resolution + start_coords[1]
        wz = cz_arr * resolution + start_coords[2]
        step_idx = np.arange(n_frames) * self.record_every

        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{"type": "scene", "rowspan": 2}, {"type": "xy"}],
                   [None,                             {"type": "xy"}]],
            column_widths=[0.6, 0.4],
            subplot_titles=[
                "Binding Trajectory (3D)",
                "Affinity Score vs Step",
                "VRMSD Convergence"
            ]
        )

        # Protein cloud
        if protein_atoms is not None and len(protein_atoms) > 0:
            pxa = [a[0] for a in protein_atoms]
            pya = [a[1] for a in protein_atoms]
            pza = [a[2] for a in protein_atoms]
            fig.add_trace(go.Scatter3d(
                x=pxa, y=pya, z=pza,
                mode="markers",
                marker=dict(size=2, color="rgba(100,200,255,0.25)"),
                name="Pocket",
                showlegend=True
            ), row=1, col=1)

        # Full trajectory path (faded)
        fig.add_trace(go.Scatter3d(
            x=wx, y=wy, z=wz,
            mode="lines",
            line=dict(color=sc_arr, colorscale="Viridis_r", width=3,
                      colorbar=dict(title="Affinity")),
            name="Trajectory Path",
            showlegend=True
        ), row=1, col=1)

        # Animated current position
        frames = []
        for i in range(n_frames):
            frames.append(go.Frame(
                data=[
                    go.Scatter3d(
                        x=[wx[i]], y=[wy[i]], z=[wz[i]],
                        mode="markers",
                        marker=dict(size=10, color="red", symbol="circle"),
                        name="Drug CoM"
                    )
                ],
                name=str(i),
                traces=[2]
            ))

        fig.add_trace(go.Scatter3d(
            x=[wx[0]], y=[wy[0]], z=[wz[0]],
            mode="markers",
            marker=dict(size=10, color="red", symbol="circle"),
            name="Drug CoM",
            showlegend=True
        ), row=1, col=1)

        # Score curve
        fig.add_trace(go.Scatter(
            x=step_idx, y=sc_arr,
            mode="lines",
            line=dict(color="cyan", width=2),
            name="Score",
            fill="tozeroy",
            fillcolor="rgba(0,200,200,0.08)"
        ), row=1, col=2)

        # VRMSD curve
        fig.add_trace(go.Scatter(
            x=step_idx, y=vr_arr,
            mode="lines",
            line=dict(color="orange", width=2),
            name="VRMSD"
        ), row=2, col=2)

        fig.update_layout(
            title=dict(
                text="KeyBox Langevin Trajectory Viewer",
                font=dict(size=18, color="white")
            ),
            paper_bgcolor="rgb(11,12,14)",
            plot_bgcolor="rgb(11,12,14)",
            font=dict(color="white"),
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                y=1.05, x=0.02,
                buttons=[
                    dict(label="Play",
                         method="animate",
                         args=[None, {"frame": {"duration": 60, "redraw": True},
                                      "fromcurrent": True, "transition": {"duration": 20}}]),
                    dict(label="Pause",
                         method="animate",
                         args=[[None], {"frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate"}])
                ]
            )],
            sliders=[dict(
                active=0,
                steps=[dict(method="animate",
                            args=[[str(i)], {"frame": {"duration": 60, "redraw": True},
                                             "mode": "immediate"}],
                            label=str(i * self.record_every))
                       for i in range(n_frames)],
                y=0, x=0.02, len=0.96,
                currentvalue=dict(prefix="Step: ", font=dict(color="white"))
            )]
        )

        fig.frames = frames
        fig.write_html(filename)
        print(f"Trajectory viewer saved: {filename}")
        print(f"  Frames: {n_frames}  Best score: {self._best_score:,.1f}")
