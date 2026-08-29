"""PNG cropping and composition helpers."""

import numpy as np
from PIL import Image as PILImage


def trim_white(path, pad_px=12, title_gap_px=8, center="title"):
    """Post-process the saved PNG: crop to the non-white content, collapse
    the white strip between the title and the figure (3-D projections
    reserve margin for rotation, so bbox_inches='tight' cannot remove it),
    and pad the narrower side so the requested anchor ends up horizontally
    centered. center='title' centers on the topmost ink band (the title);
    center='body' centers on the median x of the body ink below it, which
    is robust against an asymmetric colorbar and centers the torus itself
    (with an axes title, both coincide)."""
    img = PILImage.open(path).convert("RGB")
    a = np.asarray(img)
    nonwhite = ~(a > 250).all(axis=2)
    ys, xs = np.where(nonwhite)
    if len(ys) == 0:
        return
    rows = nonwhite.any(axis=1)
    row_ys = np.where(rows)[0]
    gap0 = np.where(~rows[row_ys[0]:])[0]
    t1 = row_ys[0] + (gap0[0] if len(gap0) else a.shape[0] - row_ys[0])
    band = nonwhite[row_ys[0]:t1]
    band_xs = np.where(band)[1]
    title_cx = 0.5 * (band_xs.min() + band_xs.max()) if len(band_xs) else \
        0.5 * (xs.min() + xs.max())
    below = np.where(rows[t1:])[0]
    body_top = t1 + (below[0] if len(below) else 0)
    # Anchor = the horizontal midpoint of the body ink extent, colorbar
    # included, so the whole figure block (torus + colorbar) is centered
    # and both side margins balance.
    cut = int(0.92 * a.shape[1])  # separates the title text from the colorbar
    body_xs = xs[ys >= body_top]
    anchor = 0.5 * (body_xs.min() + body_xs.max()) \
        if center == "body" and len(body_xs) else title_cx
    # The 3-D box is not centered in the axes window, so the title (placed
    # at the window center by matplotlib) must be shifted onto the torus
    # center in post. The shift moves the title text only; the colorbar
    # label inside the same band stays put.
    title_shift = 0
    tx0 = tx1 = 0
    if center == "body":
        band_mask = band_xs < cut
        if band_mask.any():
            tx0, tx1 = band_xs[band_mask].min(), band_xs[band_mask].max()
            title_shift = int(round(anchor - 0.5 * (tx0 + tx1)))
    # Title piece and body piece; merge when they already touch. The body
    # piece ends at its own ink bottom (the pre-crop image may carry the
    # axes window's white margin below the projection).
    pieces = [(max(row_ys[0] - pad_px, 0), t1 + pad_px),
              (max(body_top - pad_px, 0), min(ys.max() + pad_px, a.shape[0]))]
    merged = []
    for lo, hi in pieces:
        if merged and lo <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    x0 = max(xs.min() - pad_px, 0)
    x1 = min(xs.max() + pad_px, a.shape[1] - 1)
    total_h = sum(hi - lo for lo, hi in merged) + title_gap_px * (len(merged) - 1)
    canvas = PILImage.new("RGB", (x1 + 1 - x0, total_h), (255, 255, 255))
    y_cursor = 0
    for i, (lo, hi) in enumerate(merged):
        piece = img.crop((x0, lo, x1 + 1, hi))
        if i == 0 and title_shift:
            white = PILImage.new("RGB", (tx1 - tx0 + 1, piece.height),
                                 (255, 255, 255))
            piece.paste(white, (tx0 - x0, 0))
            text = img.crop((tx0, lo, tx1 + 1, hi))
            piece.paste(text, (tx0 - x0 + title_shift, 0))
        canvas.paste(piece, (0, y_cursor))
        y_cursor += piece.height + (title_gap_px if i < len(merged) - 1 else 0)
    w, h = canvas.size
    # Anchor center inside the canvas: c = anchor - x0. Padding the canvas
    # with L left / R right (L + c = (w + L + R - 1)/2) gives
    # L = w - 1 - 2c for L >= 0, else R = 2c - (w - 1).
    c = anchor - x0
    shift = int(round((w - 1) - 2.0 * c))
    out = PILImage.new("RGB", (w + abs(shift), h), (255, 255, 255))
    out.paste(canvas, (max(shift, 0), 0))
    out.save(path)


def compose_combined(path, boxes, title_box, cbar_box, pad_px=12,
                     title_gap_px=8, panel_gap_px=64, cbar_gap_px=16):
    """Post-process the combined PNG: crop each panel's axes window (given
    as saved-image boxes) and the colorbar window, tight-trim each to its
    ink, and recompose — panels side by side with a fixed gap, colorbar
    centered underneath — with the title re-centered over the composite.
    Using the known artist boxes avoids any fragile ink detection; the
    3-D boxes do not fill their axes windows, so no layout setting can
    remove the white space alone."""
    img = PILImage.open(path).convert("RGB")

    def tight(piece):
        arr = np.asarray(piece)
        nw = ~(arr > 250).all(axis=2)
        ys, xs = np.where(nw)
        if len(ys) == 0:
            return None
        y0 = max(ys.min() - pad_px, 0)
        y1 = min(ys.max() + pad_px, arr.shape[0] - 1)
        x0 = max(xs.min() - pad_px, 0)
        x1 = min(xs.max() + pad_px, arr.shape[1] - 1)
        return piece.crop((x0, y0, x1 + 1, y1 + 1))

    pieces = []
    for box in boxes:
        p = tight(img.crop((box[0], box[1], box[2] + 1, box[3] + 1)))
        if p is not None:
            pieces.append(p)
    cbar_piece = tight(img.crop((cbar_box[0], cbar_box[1],
                                 cbar_box[2] + 1, cbar_box[3] + 1)))
    title = img.crop((title_box[0], title_box[1],
                      title_box[2] + 1, title_box[3] + 1))
    # Panels row, vertically centered (the two projections have different
    # natural heights; centering balances them).
    body_w = sum(p.width for p in pieces) + panel_gap_px * (len(pieces) - 1)
    body_h = max(p.height for p in pieces)
    body = PILImage.new("RGB", (body_w, body_h), (255, 255, 255))
    x = 0
    for p in pieces:
        body.paste(p, (x, (body_h - p.height) // 2))
        x += p.width + panel_gap_px
    total_w = body_w
    total_h = body_h
    if cbar_piece is not None:
        total_w = max(total_w, cbar_piece.width)
        total_h += cbar_gap_px + cbar_piece.height
    out = PILImage.new("RGB", (total_w, title.height + title_gap_px + total_h),
                       (255, 255, 255))
    out.paste(title, ((total_w - title.width) // 2, 0))
    y = title.height + title_gap_px
    out.paste(body, ((total_w - body_w) // 2, y))
    if cbar_piece is not None:
        y += body_h + cbar_gap_px
        out.paste(cbar_piece, ((total_w - cbar_piece.width) // 2, y))
    final = PILImage.new("RGB", (out.width + 2 * pad_px, out.height + 2 * pad_px),
                         (255, 255, 255))
    final.paste(out, (pad_px, pad_px))
    final.save(path)
