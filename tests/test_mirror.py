import gzip
import hashlib
import struct

import numpy as np
import pytest

from defai import mirror


def idx_gz(array):
    array = np.asarray(array, dtype=np.uint8)
    header = struct.pack(">HBB", 0, 0x08, array.ndim) + struct.pack(f">{array.ndim}I", *array.shape)
    return gzip.compress(header + array.tobytes())


class FakeServer:
    """Stands in for urllib.request.urlopen: serves bytes by URL and counts the requests."""

    def __init__(self, files):
        self.files, self.requests = files, []

    def __call__(self, url, timeout):
        self.requests.append(url)
        server = self

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self):
                return server.files[url]

        return Response()


@pytest.fixture
def tiny_mnist(monkeypatch):
    """A 6-image 'train' split and a 4-image 'test' split registered under a test mirror."""
    rng = np.random.default_rng(0)
    images = {"train": rng.integers(0, 256, (6, 28, 28)), "test": rng.integers(0, 256, (4, 28, 28))}
    labels = {"train": np.array([0, 1, 2, 3, 4, 9]), "test": np.array([5, 6, 7, 8])}
    files = {}
    for split, (images_file, labels_file, _) in mirror.SPLIT_FILES.items():
        files[images_file], files[labels_file] = idx_gz(images[split]), idx_gz(labels[split])
    monkeypatch.setitem(mirror.MIRRORS, "test-mirror", {
        "url": "https://example.test/", "sha256": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}})
    monkeypatch.setattr(mirror, "SPLIT_FILES", {"train": (*mirror.SPLIT_FILES["train"][:2], 6), "test": (*mirror.SPLIT_FILES["test"][:2], 4)})
    return images, labels, FakeServer({"https://example.test/" + name: data for name, data in files.items()})


def test_read_idx_returns_the_stored_array():
    array = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
    assert np.array_equal(mirror.read_idx(idx_gz(array)), array)
    assert np.array_equal(mirror.read_idx(idx_gz([7, 8, 9])), [7, 8, 9])


def test_read_idx_rejects_other_formats_and_truncated_files():
    with pytest.raises(ValueError, match="unsigned bytes"):
        mirror.read_idx(gzip.compress(struct.pack(">HBB", 0, 0x0D, 1) + struct.pack(">I", 1) + b"\0\0\0\0"))  # float32 data
    with pytest.raises(ValueError, match="unsigned bytes"):
        mirror.read_idx(gzip.compress(struct.pack(">HBB", 1, 0x08, 1) + struct.pack(">I", 1) + b"\0"))  # bad magic
    with pytest.raises(ValueError, match="truncated"):
        mirror.read_idx(gzip.compress(struct.pack(">HBB", 0, 0x08, 1) + struct.pack(">I", 5) + b"\0\0"))


def test_load_split_downloads_verifies_and_caches(tiny_mnist, tmp_path):
    images, labels, server = tiny_mnist
    got_images, got_labels = mirror.load_split("test-mirror", "train", str(tmp_path), server)
    assert np.array_equal(got_images, images["train"]) and np.array_equal(got_labels, labels["train"])
    assert got_images.dtype == np.uint8 and len(server.requests) == 2
    mirror.load_split("test-mirror", "train", str(tmp_path), server)
    assert len(server.requests) == 2  # served from the cache
    assert sorted(p.name for p in (tmp_path / "test-mirror").iterdir()) == ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"]


def test_a_damaged_cache_file_is_downloaded_again(tiny_mnist, tmp_path):
    _, _, server = tiny_mnist
    name = mirror.SPLIT_FILES["test"][0]
    (tmp_path / "test-mirror").mkdir()
    (tmp_path / "test-mirror" / name).write_bytes(b"half a download")
    assert mirror.fetch("test-mirror", name, str(tmp_path), server) == server.files["https://example.test/" + name]
    assert (tmp_path / "test-mirror" / name).read_bytes() == server.files["https://example.test/" + name]


def test_a_download_with_the_wrong_checksum_is_refused_and_not_cached(tiny_mnist, tmp_path):
    _, _, server = tiny_mnist
    name = mirror.SPLIT_FILES["test"][1]
    server.files["https://example.test/" + name] = idx_gz([1, 2, 3, 4])  # valid IDX, but not the pinned file
    with pytest.raises(ValueError, match="checksum mismatch"):
        mirror.fetch("test-mirror", name, str(tmp_path), server)
    assert not (tmp_path / "test-mirror").exists()


def test_load_split_checks_the_shapes(tiny_mnist, tmp_path, monkeypatch):
    _, _, server = tiny_mnist
    monkeypatch.setattr(mirror, "SPLIT_FILES", {"train": (*mirror.SPLIT_FILES["train"][:2], 60000)})
    with pytest.raises(ValueError, match="unexpected shapes"):
        mirror.load_split("test-mirror", "train", str(tmp_path), server)


def test_the_cache_directory_can_be_chosen(monkeypatch, tmp_path):
    monkeypatch.setenv("DEFAI_DATA_DIR", str(tmp_path))
    assert mirror.cache_dir() == str(tmp_path)
    monkeypatch.delenv("DEFAI_DATA_DIR")
    assert mirror.cache_dir().endswith("/.cache/defai")


def test_the_pinned_checksums_cover_every_archive_of_every_mirror():
    wanted = {name for files in mirror.SPLIT_FILES.values() for name in files[:2]}
    for name, spec in mirror.MIRRORS.items():
        assert set(spec["sha256"]) == wanted and all(len(h) == 64 for h in spec["sha256"].values()), name
        assert spec["url"].startswith("https://") and spec["url"].endswith("/")


def test_load_split_rejects_labels_outside_the_ten_classes(tiny_mnist, tmp_path, monkeypatch):
    _, _, server = tiny_mnist
    name, url = mirror.SPLIT_FILES["train"][1], "https://example.test/" + mirror.SPLIT_FILES["train"][1]
    server.files[url] = idx_gz([0, 1, 2, 3, 4, 10])
    monkeypatch.setitem(mirror.MIRRORS["test-mirror"]["sha256"], name, hashlib.sha256(server.files[url]).hexdigest())
    with pytest.raises(ValueError, match="unexpected shapes"):
        mirror.load_split("test-mirror", "train", str(tmp_path), server)
