#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build/lightgbm-cuda-src"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"

if ! command -v git >/dev/null 2>&1; then
  echo "git is required" >&2
  exit 1
fi
if ! command -v cmake >/dev/null 2>&1; then
  echo "cmake is required" >&2
  exit 1
fi
if ! command -v ninja >/dev/null 2>&1; then
  echo "ninja is required" >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required" >&2
  exit 1
fi

COMPUTE_CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.' | tr -d ' ')"
if [[ -z "${COMPUTE_CAP}" ]]; then
  echo "Failed to detect compute capability" >&2
  exit 1
fi

echo "Detected compute capability: ${COMPUTE_CAP}"
rm -rf "${BUILD_DIR}"
git clone --branch v4.6.0 --depth 1 --recursive https://github.com/microsoft/LightGBM.git "${BUILD_DIR}"

python3 - <<'PY' "${BUILD_DIR}/CMakeLists.txt" "${BUILD_DIR}/python-package/pyproject.toml" "${COMPUTE_CAP}"
from pathlib import Path
import sys
cmake_path = Path(sys.argv[1])
pyproject_path = Path(sys.argv[2])
cap = sys.argv[3]
text = cmake_path.read_text()
old = '''    # reference for mapping of CUDA toolkit component versions to supported architectures ("compute capabilities"):
    # https://en.wikipedia.org/wiki/CUDA#GPUs_supported
    set(CUDA_ARCHS "60" "61" "62" "70" "75")
    if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.0")
        list(APPEND CUDA_ARCHS "80")
    endif()
    if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.1")
        list(APPEND CUDA_ARCHS "86")
    endif()
    if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.5")
        list(APPEND CUDA_ARCHS "87")
    endif()
    if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.8")
        list(APPEND CUDA_ARCHS "89")
        list(APPEND CUDA_ARCHS "90")
    endif()
    if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.8")
        list(APPEND CUDA_ARCHS "100")
        list(APPEND CUDA_ARCHS "120")
    endif()
    # Generate PTX for the most recent architecture for forwards compatibility
    list(POP_BACK CUDA_ARCHS CUDA_LAST_SUPPORTED_ARCH)
    list(TRANSFORM CUDA_ARCHS APPEND "-real")
    list(APPEND CUDA_ARCHS "${CUDA_LAST_SUPPORTED_ARCH}-real" "${CUDA_LAST_SUPPORTED_ARCH}-virtual")
'''
new = f'''    # Limit build to the detected GPU architecture to avoid unsupported legacy targets.
    set(CUDA_ARCHS "{cap}-real" "{cap}-virtual")
'''
if old not in text:
    raise SystemExit("Failed to patch CMakeLists.txt")
cmake_path.write_text(text.replace(old, new))
py_text = pyproject_path.read_text()
needle = 'cmake.args = [\n    "-D__BUILD_FOR_PYTHON:BOOL=ON"\n]\n'
replacement = 'cmake.args = [\n    "-D__BUILD_FOR_PYTHON:BOOL=ON"\n]\ncmake.source-dir = ".."\n'
if needle not in py_text:
    raise SystemExit("Failed to patch pyproject.toml")
pyproject_path.write_text(py_text.replace(needle, replacement))
print(f"Patched CUDA_ARCHS to {cap} and set cmake.source-dir=..")
PY

cp "${BUILD_DIR}/LICENSE" "${BUILD_DIR}/python-package/LICENSE"
cd "${BUILD_DIR}/python-package"
export CUDACXX="${CUDA_HOME}/bin/nvcc"
export CUDA_PATH="${CUDA_HOME}"
export CMAKE_ARGS="-DUSE_CUDA=ON"
uv pip install --python "${PYTHON_BIN}" --upgrade scikit-build-core
uv pip install --python "${PYTHON_BIN}" --reinstall --no-deps .

echo "LightGBM CUDA build installed into ${ROOT_DIR}/.venv"
