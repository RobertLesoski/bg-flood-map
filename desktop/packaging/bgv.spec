# PyInstaller spec: one-folder Windows app "BG Flood Viewer". Run via build.ps1 (cwd = desktop/packaging).
import os
from PyInstaller.utils.hooks import collect_all, collect_submodules

HERE = os.path.abspath(SPECPATH)
DESKTOP = os.path.dirname(HERE)
import sys
sys.path.insert(0, DESKTOP)  # so collect_submodules("bgv") sees the package (layers are imported by name at runtime)

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
# venv built from an Anaconda base: stdlib extension DLL deps (_ctypes->ffi, _ssl, _lzma, _bz2, _sqlite3, pyexpat)
# live in <base>\Library\bin, which PyInstaller does not search.
import sys, glob
_libbin = os.path.join(sys.base_prefix, "Library", "bin")
for pat in ("ffi*.dll", "libssl-*.dll", "libcrypto-*.dll", "liblzma.dll", "libbz2.dll", "sqlite3.dll",
            "libexpat.dll", "zlib*.dll"):
    binaries += [(f, ".") for f in glob.glob(os.path.join(_libbin, pat))]
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
