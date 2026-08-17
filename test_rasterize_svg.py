import numpy as np
import pytest

from rasterize_svg import rasterize_svg


def make_svg(tmp_path, content, width=20, height=20):
    path = tmp_path / "shape.svg"
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}">{content}</svg>'
    )
    return str(path)


def test_output_shape_and_dtype(tmp_path):
    svg = make_svg(tmp_path, '<rect width="20" height="20" fill="black"/>')
    arr = rasterize_svg(svg, 20, 20, 128)
    assert arr.shape == (20, 20)
    assert arr.dtype == bool


def test_output_shape_is_height_by_width(tmp_path):
    svg = make_svg(tmp_path, '<rect width="20" height="20" fill="black"/>')
    arr = rasterize_svg(svg, width=30, height=10, threshold=128)
    assert arr.shape == (10, 30)


def test_solid_black_fill_is_all_live(tmp_path):
    svg = make_svg(tmp_path, '<rect width="20" height="20" fill="black"/>')
    arr = rasterize_svg(svg, 20, 20, 128)
    assert arr.all()


def test_solid_white_fill_is_all_dead(tmp_path):
    svg = make_svg(tmp_path, '<rect width="20" height="20" fill="white"/>')
    arr = rasterize_svg(svg, 20, 20, 128)
    assert not arr.any()


def test_empty_svg_is_all_dead(tmp_path):
    svg = make_svg(tmp_path, "")
    arr = rasterize_svg(svg, 20, 20, 128)
    assert not arr.any()


def test_rect_renders_at_correct_position_and_size(tmp_path):
    svg = make_svg(tmp_path, '<rect x="5" y="5" width="10" height="10" fill="black"/>')
    arr = rasterize_svg(svg, 20, 20, 128)

    expected = np.zeros((20, 20), dtype=bool)
    expected[5:15, 5:15] = True
    np.testing.assert_array_equal(arr, expected)


def test_circle_shape_is_live_at_center_and_dead_at_corners(tmp_path):
    svg = make_svg(tmp_path, '<circle cx="10" cy="10" r="5" fill="black"/>')
    arr = rasterize_svg(svg, 20, 20, 128)

    assert arr[10, 10]
    assert not arr[0, 0]
    assert not arr[0, 19]
    assert not arr[19, 0]
    assert not arr[19, 19]


@pytest.mark.parametrize(
    "fill, threshold, expect_live",
    [
        ("rgb(50,50,50)", 100, True),   # dark gray, below threshold
        ("rgb(200,200,200)", 100, False),  # light gray, above threshold
        ("rgb(150,150,150)", 10, False),   # mid gray, low threshold excludes it
        ("rgb(150,150,150)", 250, True),   # mid gray, high threshold includes it
    ],
)
def test_threshold_controls_which_gray_levels_are_live(
    tmp_path, fill, threshold, expect_live
):
    svg = make_svg(tmp_path, f'<rect width="20" height="20" fill="{fill}"/>')
    arr = rasterize_svg(svg, 20, 20, threshold)
    if expect_live:
        assert arr.all()
    else:
        assert not arr.any()
