# Frontend Stage Renderer Conventions

- Use `rendererType: "mask"` for binary spatial inclusion/exclusion artifacts.
  Examples: selected surface masks, plane inliers, suppression masks, threshold masks, ROI masks.

- Use `rendererType: "image"` for scalar rasters and heatmaps.
  Examples: residual heatmaps, depth gradients, normalized height previews, raw height previews.

- Use `rendererType: "overlay"` for vector-like or object-centric annotation layers.
  Examples: contours, labels, connected-component overlays, classification overlays.

- Keep stage-specific artifact matching and reference preference in stage semantics and mask-spec helpers.
  `MaskArtifactViewer` should stay generic and only own reusable mask viewing behavior.
