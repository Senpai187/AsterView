"""Compare reference and native plane reconstruction on identical geometry."""
import json
import platform
import statistics
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from asterview.model import Volume
from asterview.fast_reslice import FastReslicer
from asterview.reformat import sample_plane, rotation_matrix, base_frame


def measure(fn, count=15):
    fn()
    values = []
    for _ in range(count):
        start = time.perf_counter()
        fn()
        values.append((time.perf_counter() - start) * 1000)
    return {"median_ms": round(statistics.median(values),3), "p95_ms":round(float(np.percentile(values,95)),3), "samples":count}


def main():
    output = Path(sys.argv[1]) if len(sys.argv)>1 else Path('navigation-benchmark.json')
    data = np.broadcast_to(np.linspace(-1000, 2500, 512, dtype=np.float32)[None,None,:],(512,512,512)).copy()
    volume = Volume(data, (.4,.4,.4), np.zeros(3), np.eye(3))
    rotation = rotation_matrix(17,21,12)
    frames = [base_frame(name,rotation) for name in ('Axial','Coronal','Sagittal')]
    engine = FastReslicer(volume)
    right,up = frames[0]
    kwargs = dict(center=volume.center,right=right,up=up,size_mm=(200,200),shape=(512,512))
    reference = sample_plane(volume,**kwargs)
    native = engine.sample_plane(**kwargs)
    assert np.max(np.abs(reference-native)) < .01
    result = {"version":"0.2.0", "date":"2026-10-20", "environment":{"python":platform.python_version(),"platform":platform.platform()},"source":{"shape":[512,512,512],"spacing_mm":[.4,.4,.4],"size_MiB":512},"limitations":"Synthetic analytic volume. Reconstruction timings exclude DICOM I/O, first GPU upload, Qt event dispatch and display presentation. Slabs/padding use the validated reference fallback.","max_intensity_difference":float(np.max(np.abs(reference-native))),"reference_plane":measure(lambda:sample_plane(volume,**kwargs)),"native_plane":measure(lambda:engine.sample_plane(**kwargs))}
    def triplet(shape):
        for right,up in frames:
            engine.sample_plane(volume.center,right,up,size_mm=(200,200),shape=shape)
    result['native_three_planes_full'] = measure(lambda:triplet((512,512)))
    result['native_three_planes_preview'] = measure(lambda:triplet((224,224)))
    result['plane_speedup_median'] = round(result['reference_plane']['median_ms']/result['native_plane']['median_ms'],2)
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
