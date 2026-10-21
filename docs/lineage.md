# Lineage

- **Research name:** "VTL spatial compliance" / the VTL Spatial Kernel. It began as a Colab notebook running the Visual Thinking Lens gradient-field kernel on robot scenes ("Gemini Robotics-ER 1.6 — VTL Kernel Metrics"), then became the `vtl_spatial` package. Here it's `spatial_kernel`. Output JSON keeps its original `schema_version` strings (`vtl-spatial-compliance/v1`, `vtl-parallax-success/v1`) so recorded reports stay comparable.
- **Code:** the package and benchmarks are the research code as of August 2026. Changes for this repository: the package name; the CLI name; benchmark input/output folders (`data/`, `results/`); the two-view functions moved from `benchmark/realframe_parallax.py` into `spatial_kernel/twoview.py` unchanged, with a `check_pair` wrapper and a `twoview` CLI command; relative paths in placement output. The real-photo results reproduce byte for byte after these changes.
- **Paper:** [The VTL Spatial Kernel](https://www.parallaxmetrology.com/SPATIAL_KERNEL_PAPER.html) (2026). The site page: [spatial_pm.html](https://www.parallaxmetrology.com/spatial_pm.html).
- **Independence of reads (Ω):** borrowed from the Parallax seismology work, where agreement among physically independent reads, not transforms of one signal, is what counts.
- **Not here:** the authority-separated auditor and its studies (see the README).
