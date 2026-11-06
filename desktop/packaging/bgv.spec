# PyInstaller spec: one-folder Windows app "BG Flood Viewer". Run via build.ps1 (cwd = desktop/packaging).
import os
from PyInstaller.utils.hooks import collect_all, collect_submodules

HERE = os.path.abspath(SPECPATH)
DESKTOP = os.path.dirname(HERE)

datas, binaries, hidden = [], [], []
# wgpu: wgpu-native dll + shaders/resources; imgui_bundle: fonts/assets; rasterio: gdal_data + proj_data (proj.db);
# rendercanvas/glfw: glfw3.dll; imageio_ffmpeg: ffmpeg exe (cinematic export); PIL plugins.
for pkg in ("wgpu", "imgui_bundle", "rendercanvas", "glfw", "rasterio", "pyproj", "imageio_ffmpeg", "PIL",
            "shapely", "mapbox_earcut", "numba", "llvmlite"):
    try:
        d, b, h = collect_all(pkg)
    except Exception as e:
        print("collect_all skipped", pkg, e)
        continue
    datas += d; binaries += b; hidden += h
hidden += collect_submodules("bgv") + ["rasterio.sample", "rasterio.vrt", "rasterio._features", "scipy.special._cdflib"]
hidden = [h for h in hidden if ".tests" not in h and not h.endswith(".tests")]

a = Analysis([os.path.join(HERE, "bgv_entry.py")], pathex=[DESKTOP], binaries=binaries, datas=datas,
             hiddenimports=hidden, excludes=["pytest", "matplotlib", "IPython", "jupyter", "notebook", "tkinter.test",
                                             "PyQt5", "PySide6", "pandas.tests", "numpy.tests", "scipy.tests"],
             noarchive=False)
a.datas = [x for x in a.datas if "/tests/" not in x[0].replace("\\", "/")]
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="BG Flood Viewer", console=False,
          icon=os.path.join(HERE, "bgv.ico"), upx=False)
coll = COLLECT(exe, a.binaries, a.datas, name="BG Flood Viewer", upx=False)
