import io

import cairosvg
import numpy as np
from PIL import Image


def rasterize_svg(svg_path, width, height, threshold):
    """Rasterize an SVG file into a boolean numpy array.

    Renders the SVG at the given pixel dimensions, then thresholds it into
    a live/dead grid suitable for seeding a Game of Life pattern: a pixel is
    "on" if it is opaque and its luminance falls below `threshold`.

    Args:
        svg_path: Path to the source .svg file.
        width: Output raster width in pixels.
        height: Output raster height in pixels.
        threshold: Luminance cutoff in [0, 255]. Pixels darker than this
            (and not fully transparent) are considered live cells.

    Returns:
        A (height, width) boolean numpy array.
    """
    png_bytes = cairosvg.svg2png(
        url=svg_path, output_width=width, output_height=height
    )
    image = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    rgba = np.array(image)

    rgb = rgba[..., :3].astype(np.float64)
    alpha = rgba[..., 3]
    luminance = rgb @ np.array([0.2126, 0.7152, 0.0722])

    return (alpha > 0) & (luminance < threshold)
