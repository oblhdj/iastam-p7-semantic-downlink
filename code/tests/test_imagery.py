from pathlib import Path

import cv2
import numpy as np
import pytest

from sat7.imagery import decode_image, load_image
from sat7.scheduler import RAW_TILE_BYTES

QS_TILE = Path(__file__).resolve().parents[2] / "demo" / "quickstart" / "tiles" / "01_synthetic_00113a75c.jpg"


def _jpg(img) -> bytes:
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def test_tile_dimensions_and_both_sizes(tmp_path):
    img = np.random.default_rng(0).integers(0, 255, (768, 768, 3), np.uint8)
    p = tmp_path / "t.jpg"
    p.write_bytes(_jpg(img))
    out, info = load_image(p)
    assert out.shape == (768, 768, 3) and out.dtype == np.uint8
    assert (info.width, info.height, info.format) == (768, 768, "JPEG")
    assert info.file_bytes == p.stat().st_size                 # compressed, on disk
    assert info.raw_bytes == RAW_TILE_BYTES                    # the B0 definition, not the file size


def test_pixels_match_the_repo_loader():
    if not QS_TILE.exists():
        pytest.skip("quickstart tiles not present")
    out, _ = load_image(QS_TILE)
    assert np.array_equal(out, cv2.imread(str(QS_TILE), cv2.IMREAD_COLOR))


def test_grey_and_alpha_become_three_channels():
    g, info = decode_image(cv2.imencode(".png", np.full((40, 50), 90, np.uint8))[1].tobytes(), "g.png")
    assert g.shape == (40, 50, 3) and info.channels_in == 1 and info.notes
    a, info = decode_image(cv2.imencode(".png", np.full((40, 50, 4), 90, np.uint8))[1].tobytes(), "a.png")
    assert a.shape == (40, 50, 3) and info.channels_in == 4 and info.notes


@pytest.mark.parametrize("data,name,msg", [
    (b"", "e.jpg", "empty"),
    (b"not an image at all", "x.jpg", "not a supported image"),
    (b"\xff\xd8\xff\xe0" + b"\x00" * 50, "t.jpg", "does not decode"),
])
def test_bad_bytes_are_rejected(data, name, msg):
    with pytest.raises(ValueError, match=msg):
        decode_image(data, name)


def test_sixteen_bit_is_rejected_not_stretched():
    data = cv2.imencode(".png", np.full((64, 64, 3), 4000, np.uint16))[1].tobytes()
    with pytest.raises(ValueError, match="uint16"):
        decode_image(data, "deep.png")


def test_too_small_is_rejected():
    with pytest.raises(ValueError, match="minimum"):
        decode_image(_jpg(np.zeros((8, 8, 3), np.uint8)), "tiny.jpg")


def test_extension_checks(tmp_path):
    p = tmp_path / "a.gif"
    p.write_bytes(b"GIF89a")
    with pytest.raises(ValueError, match="unsupported extension"):
        load_image(p)
    with pytest.raises(FileNotFoundError):
        load_image(tmp_path / "missing.jpg")
    _, info = decode_image(_jpg(np.zeros((32, 32, 3), np.uint8)), "mislabelled.png")
    assert info.format == "JPEG" and any("extension" in n for n in info.notes)
