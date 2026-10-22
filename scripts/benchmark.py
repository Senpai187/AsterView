"""Reproducible synthetic timing, not a scanner or clinical validation study."""
import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PySide6.QtWidgets import QApplication
from asterview.model import Volume
from asterview.demo import make_demo
from asterview.reformat import sample_plane, rotation_matrix, base_frame
from asterview.render3d import VolumeView


def timings(function, n=9):
    function()
    values = []
    for _ in range(n):
        start = time.perf_counter()
        function()
        values.append((time.perf_counter() - start) * 1000)
    return {"median_ms": round(statistics.median(values), 2), "p95_ms": round(float(np.percentile(values, 95)), 2), "samples": n}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("benchmark-results.json"))
    args = parser.parse_args()
    data = np.broadcast_to(np.linspace(-1000, 2500, 512, dtype=np.float32)[None, None, :], (512, 512, 512)).copy()
    volume = Volume(data, (.4,.4,.4), np.zeros(3), np.eye(3))
    rotation = rotation_matrix(17, 21, 12)
    right, up = base_frame("Axial", rotation)
    def plane(slab=0):
        return sample_plane(volume, volume.center, right, up, size_mm=(200,200), shape=(512,512), slab_mm=slab)
    result = {"environment":{"python":platform.python_version(),"platform":platform.platform()}, "limitations":"Synthetic data only. Timings exclude DICOM disk I/O, codec decoding, UI event dispatch and real scanner validation.", "source":{"shape":[512,512,512],"MiB":512,"spacing_mm":[.4,.4,.4]}, "oblique_512x512_plane":timings(plane), "oblique_512x512_5mm_mean_slab":timings(lambda:plane(5),5)}
    app = QApplication.instance() or QApplication([])
    widget = VolumeView()
    widget.resize(640, 480)
    widget.show()
    widget.set_volume(make_demo())
    for _ in range(5):
        app.processEvents()
    def render():
        widget.renderer.GetActiveCamera().Azimuth(2)
        widget._window.Render()
    result["gpu_demo_640x480"] = timings(render)
    result["gpu_demo_640x480"]["renderer"] = widget.info
    result["gpu_demo_640x480"]["source_shape"] = [160,192,192]
    widget.cleanup()
    widget.close()
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
