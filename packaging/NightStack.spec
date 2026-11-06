# PyInstaller spec - builds NightStack.app for Intel (x86_64) Macs.
# Run via ./build_mac.sh  (or: pyinstaller --noconfirm packaging/NightStack.spec)
import os
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
ICNS = os.path.join(ROOT, "build", "NightStack.icns")

datas = [(os.path.join(ROOT, "nightstack", "resources"), "nightstack/resources")]
datas += collect_data_files("astropy", includes=["**/*.cfg", "**/*.dat", "**/*.json"])
binaries = collect_dynamic_libs("rawpy")

a = Analysis(
    [os.path.join(ROOT, "main.py")],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=["rawpy", "rawpy._rawpy", "scipy.ndimage", "tifffile", "astropy.io.fits", "certifi"],
    excludes=["tkinter", "matplotlib", "IPython", "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets",
              "PySide6.Qt3DCore", "PySide6.QtQuick", "PySide6.QtQml", "PySide6.QtMultimedia",
              "PySide6.QtCharts", "PySide6.QtDataVisualization", "PySide6.QtPdf"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="NightStack", debug=False, strip=False,
          upx=False, console=False, target_arch="x86_64")
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="NightStack")
app = BUNDLE(
    coll,
    name="NightStack.app",
    icon=ICNS if os.path.exists(ICNS) else None,
    bundle_identifier="app.nightstack.NightStack",
    info_plist={
        "CFBundleName": "NightStack",
        "CFBundleDisplayName": "NightStack",
        "CFBundleShortVersionString": "1.3.0",
        "CFBundleVersion": "1.3.0",
        "LSMinimumSystemVersion": "11.0",
        "NSHighResolutionCapable": True,
        "LSApplicationCategoryType": "public.app-category.photography",
        "CFBundleDocumentTypes": [{
            "CFBundleTypeName": "Image",
            "CFBundleTypeRole": "Viewer",
            "LSHandlerRank": "Alternate",
            "LSItemContentTypes": ["public.image", "public.camera-raw-image", "public.tiff", "public.jpeg",
                                   "public.png"],
        }],
    },
)
