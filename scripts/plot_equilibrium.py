#!/usr/bin/env python3
"""Single command-line entry point for the cuMES plotting package."""

from cumes_plot.config import FigureConfig


# ---------------------------------------------------------------------------
# User-adjustable figure parameters
#
# Keep actual rendering choices here. Implementation functions live in the
# cumes_plot package and receive this object explicitly.
# ---------------------------------------------------------------------------

FIGURE_PARAMETERS = FigureConfig(
    panel_rows=2,
    panel_columns=3,
    coordinate_figure_size=(10.2, 6.4),
    field_slice_figure_size=(10.8, 6.4),
    field_contour_figure_size=(10.8, 6.5),
    save_dpi=300,
    save_face_color="white",
    save_bbox="tight",
    share_panel_x=True,
    share_panel_y=True,
    use_constrained_layout=True,
    # Coordinate-mesh density and style. Theta lines run radially across R-Z.
    coordinate_radial_line_count=9,
    coordinate_theta_line_count=24,
    coordinate_flux_color="#17365d",
    coordinate_theta_color="#b33b2e",
    coordinate_inner_flux_linewidth=0.8,
    coordinate_edge_flux_linewidth=1.35,
    coordinate_theta_linewidth=0.65,
    coordinate_theta_alpha=0.9,
    # Magnetic-field colors and contours.
    field_colormap="viridis",
    field_shading="gouraud",
    field_rasterized=True,
    field_contour_level_count=17,
    field_contour_line_stride=2,
    field_contour_line_color="black",
    field_contour_linewidth=0.28,
    field_contour_line_alpha=0.55,
    field_contour_surface_count=6,
    field_contour_inner_fraction=0.08,
    # Shared axes, typography, and output settings.
    slice_limit_surface_count=9,
    slice_limit_margin_fraction=0.04,
    slice_limit_zero_span_margin=0.1,
    panel_title_fontsize=10,
    tick_label_fontsize=8,
    axis_label_fontsize=9,
    figure_title_fontsize=12,
    colorbar_shrink=0.88,
    colorbar_pad=0.02,
    colorbar_label=r"$|B|$ (T)",
    slice_aspect="equal",
    slice_aspect_adjustable="box",
    normalized_angle_limits=(0.0, 1.0),
    field_contour_extend="both",
    figure_title_prefix="cuMES magnetic coordinates",
    min_render_theta=120,
    min_render_zeta=120,
    # 3-D overview figures and backend-neutral scene styling.
    overview_single_figure_size=(7.4, 5.6),
    overview_combined_figure_size=(8.8, 5.4),
    overview_slices_figure_size=(10.0, 5.6),
    overview_working_dpi=100,
    overview_save_dpi=300,
    pyvista_save_dpi=220,
    pyvista_window_size=(1400, 1050),
    overview_title_fontsize=11,
    overview_colorbar_label="|B| (T)",
    overview_colorbar_label_fontsize=10,
    overview_colorbar_tick_fontsize=9,
    light_azimuth_degrees=330,
    light_altitude_degrees=55,
    surface_intensity_floor=0.55,
    coil_color="#CD7F32",
    curve_color="#0b0b0b",
    field_line_width=1.1,
    field_line_alpha=0.85,
    axis_line_width=1.4,
    perspective_view=(24.0, -58.0),
    top_view=(90.0, -90.0),
    field_line_seed_fractions=(0.0, 1.0 / 3.0, 2.0 / 3.0),
    coil_default_radius_fraction=0.006,
    coil_visible_extent_factor=50.0,
    horizontal_limit_margin=1.06,
    vertical_limit_margin=1.10,
)


def main():
    from cumes_plot.application import main as run_application

    run_application(FIGURE_PARAMETERS)


if __name__ == "__main__":
    main()
