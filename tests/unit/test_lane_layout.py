import pathlib
import sys
import json

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from edge.app.services.algorithms.lane_layout import build_lane_shapes, resolve_lane_by_point  # noqa: E402


def test_lane_polygons_build_and_resolve():
    polygons = (
        "1:40-60|290-60|250-620|60-620;"
        "2:330-60|580-60|540-620|350-620;"
        "3:700-60|950-60|910-620|720-620;"
        "4:990-60|1240-60|1200-620|1010-620"
    )
    shapes = build_lane_shapes(
        frame_width=1280,
        frame_height=640,
        target_lanes=[1, 2, 3, 4],
        lane_polygons_text=polygons,
    )
    assert len(shapes) == 4
    assert shapes[0]["kind"] == "polygon"

    assert resolve_lane_by_point(
        x=120,
        y=300,
        frame_width=1280,
        frame_height=640,
        target_lanes=[1, 2, 3, 4],
        lane_polygons_text=polygons,
    ) == 1
    assert resolve_lane_by_point(
        x=430,
        y=300,
        frame_width=1280,
        frame_height=640,
        target_lanes=[1, 2, 3, 4],
        lane_polygons_text=polygons,
    ) == 2


def test_lane_layout_file_overrides_inline_text(tmp_path):
    layout_path = tmp_path / "lane_layout.json"
    layout_path.write_text(
        json.dumps(
            {
                "frame_width": 1280,
                "frame_height": 640,
                "lanes": [
                    {"lane": 1, "points": [[0, 0], [200, 0], [200, 639], [0, 639]]},
                    {"lane": 2, "points": [[200, 0], [400, 0], [400, 639], [200, 639]]},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    lane = resolve_lane_by_point(
        x=120,
        y=300,
        frame_width=1280,
        frame_height=640,
        target_lanes=[1, 2],
        lane_layout_file=str(layout_path),
        lane_polygons_text="1:500-0|700-0|700-639|500-639",
    )
    assert lane == 1
