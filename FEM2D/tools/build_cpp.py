from pathlib import Path
import shutil
import subprocess
import sys


root = Path(__file__).resolve().parents[1]
cpp_dir = root / "cpp"
build_dir = root / "build" / "cpp"
output_dir = root / "python" / "FEM2D"

build_dir.mkdir(parents=True, exist_ok=True)

cmake = shutil.which("cmake") or "/opt/homebrew/bin/cmake"

pybind11_cmakedir = subprocess.check_output(
    [sys.executable, "-m", "pybind11", "--cmakedir"],
    text=True,
).strip()

cmd_configure = [
    cmake,
    "-S", str(cpp_dir),
    "-B", str(build_dir),
    f"-DPython_EXECUTABLE={sys.executable}",
    f"-Dpybind11_DIR={pybind11_cmakedir}",
]

cmd_build = [
    cmake,
    "--build", str(build_dir),
]

print(f"Project root : {root}")
print(f"CMake        : {cmake}")
print(f"Python       : {sys.executable}")
print(f"pybind11_DIR : {pybind11_cmakedir}")
print(f"Output dir   : {output_dir}")

print("\nCONFIGURE:")
subprocess.check_call(cmd_configure)

print("\nBUILD:")
subprocess.check_call(cmd_build)

print("\nBUILT EXTENSIONS:")
for path in sorted(output_dir.glob("_refine*")):
    print(path)

print("\nDONE")