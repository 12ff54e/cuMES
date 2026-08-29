"""Typed configuration passed from the plotting entry point to renderers."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FigureConfig:
    """All user-adjustable parameters shared by the plotting package."""

    panel_rows: int
    panel_columns: int
    coordinate_figure_size: tuple
    field_slice_figure_size: tuple
    field_contour_figure_size: tuple
    save_dpi: int
    save_face_color: str
    save_bbox: str
    share_panel_x: bool
    share_panel_y: bool
    use_constrained_layout: bool
    coordinate_radial_line_count: int
    coordinate_theta_line_count: int
    coordinate_flux_color: str
    coordinate_theta_color: str
    coordinate_inner_flux_linewidth: float
    coordinate_edge_flux_linewidth: float
    coordinate_theta_linewidth: float
    coordinate_theta_alpha: float
    field_colormap: str
    field_shading: str
    field_rasterized: bool
    field_contour_level_count: int
    field_contour_line_stride: int
    field_contour_line_color: str
    field_contour_linewidth: float
    field_contour_line_alpha: float
    field_contour_surface_count: int
    field_contour_inner_fraction: float
    slice_limit_surface_count: int
    slice_limit_margin_fraction: float
    slice_limit_zero_span_margin: float
    panel_title_fontsize: float
    tick_label_fontsize: float
    axis_label_fontsize: float
    figure_title_fontsize: float
    colorbar_shrink: float
    colorbar_pad: float
    colorbar_label: str
    slice_aspect: str
    slice_aspect_adjustable: str
    normalized_angle_limits: tuple
    field_contour_extend: str
    figure_title_prefix: str
    min_render_theta: int
    min_render_zeta: int
    overview_single_figure_size: tuple
    overview_combined_figure_size: tuple
    overview_slices_figure_size: tuple
    overview_working_dpi: int
    overview_save_dpi: int
    pyvista_save_dpi: int
    pyvista_window_size: tuple
    overview_title_fontsize: float
    overview_colorbar_label: str
    overview_colorbar_label_fontsize: float
    overview_colorbar_tick_fontsize: float
    light_azimuth_degrees: float
    light_altitude_degrees: float
    surface_intensity_floor: float
    coil_color: str
    curve_color: str
    field_line_width: float
    field_line_alpha: float
    axis_line_width: float
    perspective_view: tuple
    top_view: tuple
    field_line_seed_fractions: tuple
    coil_default_radius_fraction: float
    coil_visible_extent_factor: float
    horizontal_limit_margin: float
    vertical_limit_margin: float

    @property
    def slice_panel_count(self):
        return self.panel_rows * self.panel_columns


def validate_figure_config(config):
    """Reject inconsistent configuration before starting expensive work."""
    if config.panel_rows < 1 or config.panel_columns < 1:
        raise ValueError("figure panel dimensions must be positive")
    if config.slice_panel_count != 6:
        raise ValueError("coordinate figures currently require six panels")
    counts = (
        config.coordinate_radial_line_count,
        config.coordinate_theta_line_count,
        config.field_contour_level_count,
        config.field_contour_line_stride,
        config.field_contour_surface_count,
        config.slice_limit_surface_count,
        config.min_render_theta,
        config.min_render_zeta,
        config.save_dpi,
    )
    if any(value < 1 for value in counts):
        raise ValueError("figure counts, sampling, and DPI must be positive")
    if not 0.0 <= config.field_contour_inner_fraction < 1.0:
        raise ValueError("field contour inner fraction must lie in [0, 1)")
    if config.normalized_angle_limits[0] >= \
            config.normalized_angle_limits[1]:
        raise ValueError("normalized angle limits must increase")
    overview_counts = (
        config.overview_working_dpi,
        config.overview_save_dpi,
        config.pyvista_save_dpi,
    )
    if any(value < 1 for value in overview_counts):
        raise ValueError("overview figure DPI values must be positive")
    if any(value <= 0.0 for size in (
            config.coordinate_figure_size, config.field_slice_figure_size,
            config.field_contour_figure_size,
            config.overview_single_figure_size,
            config.overview_combined_figure_size,
            config.overview_slices_figure_size) for value in size):
        raise ValueError("figure dimensions must be positive")
    if any(not 0.0 <= fraction < 1.0
           for fraction in config.field_line_seed_fractions):
        raise ValueError("field-line seed fractions must lie in [0, 1)")
    if config.coil_default_radius_fraction <= 0.0 or \
            config.coil_visible_extent_factor <= 0.0 or \
            config.horizontal_limit_margin <= 0.0 or \
            config.vertical_limit_margin <= 0.0:
        raise ValueError("coil and view-scale parameters must be positive")
    return config


# Compatibility name for callers that used the first package revision.
CoordinateFigureConfig = FigureConfig
validate_coordinate_figure_config = validate_figure_config
