# Sphere consistency feature family

Raw visible-surface sphere-fit RMSE (`surface_sphere_fit_rmse_mm`) remains available for diagnostics, but it is a poor primary ballness signal on its own:

- It is **scale-dependent** (the same geometric error looks smaller on larger objects).
- It is computed over the **visible 2.5D cap only**, not a complete 3D scan.
- Different superclasses (BALL_GOOD, BALL_SCRAP, SCRAP_METAL) can overlap strongly in absolute RMSE histograms.

Use the **sphere-consistency family** as a consensus instead:

| Feature | Interpretation |
| --- | --- |
| `surface_sphere_fit_rmse_norm` | Lower = tighter sphere fit relative to object size |
| `surface_sphere_fit_residual_p95_norm` | Lower = tail residuals stay modest |
| `surface_sphere_fit_residual_mad_norm` | Lower = robust residual dispersion is tight |
| `surface_sphere_radius_error_norm` | Lower = fitted sphere radius matches expected diameter |
| `surface_visible_cap_fraction` | Higher = fuller visible spherical cap |
| `surface_volume_fill_ratio` | Higher = measured volume matches expected sphere/cap volume |
| `surface_sphere_fit_confidence` | Higher = sphere-fit diagnostics are more trustworthy |

Diagnostic-only companions:

- `surface_sphere_center_depth_ratio` — null when expected center depth is unreliable
- `surface_sphere_vs_ellipsoid_gain` — high values suggest ellipsoid fits much better than a sphere

Ball-like classification should combine normalized residuals, radius plausibility, cap/volume evidence, axis balance, footprint roundness, roughness/discontinuity, and fit confidence — not raw RMSE alone.
