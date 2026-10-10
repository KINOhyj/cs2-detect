"""
一键编译 overlay.cpp 为 overlay.pyd
用法：python build.py
"""
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

# ---- 配置 ----
SOURCE     = Path(__file__).parent / "overlay.cpp"
MODULE     = "overlay"                      # 生成 overlay.pyd
EXTRA_LIBS = ["user32.lib", "gdi32.lib", "dwmapi.lib", "winmm.lib"]
OPT_FLAGS  = ["/O2", "/std:c++17", "/EHsc", "/LD", "/MD", "/DNDEBUG"]


class BuildError(RuntimeError):
    """编译 overlay 扩展失败。"""


def find_pybind11_include() -> str:
    """返回 pybind11 的头文件目录。

    Raises:
        BuildError: 未安装 pybind11
    """
    try:
        import pybind11
    except ImportError:
        raise BuildError("未安装 pybind11，请先运行：pip install pybind11")
    return pybind11.get_include()


def find_vcvars() -> Path | None:
    """定位 Visual Studio 的 vcvars64.bat；找不到返回 None。"""
    candidates = [
        r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\VC\Auxiliary\Build\vcvars64.bat",
    ]
    for p in candidates:
        if Path(p).exists():
            return Path(p)
    # 尝试用 vswhere 自动查找
    vswhere = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe")
    if vswhere.exists():
        out = subprocess.run(
            [str(vswhere), "-latest", "-property", "installationPath"],
            capture_output=True, text=True
        ).stdout.strip()
        if out:
            p = Path(out) / "VC/Auxiliary/Build/vcvars64.bat"
            if p.exists():
                return p
    return None


def build() -> Path:
    """编译 overlay.cpp，返回生成的 .pyd 路径。

    作为库函数调用（如 main.py 的 load_overlay）时不会终止进程，
    失败统一抛 BuildError，由调用方决定怎么处理。

    Raises:
        BuildError: 源文件缺失 / 缺 pybind11 / 缺 MSVC / 编译返回非零
    """
    if not SOURCE.exists():
        raise BuildError(f"找不到源文件：{SOURCE}")

    py_include = sysconfig.get_paths()["include"]
    py_lib     = Path(py_include).parent / "libs"
    pybind_inc = find_pybind11_include()
    output     = Path(__file__).parent / f"{MODULE}.pyd"

    # 组 cl 命令
    cl_args = [
        "cl",
        *OPT_FLAGS,
        str(SOURCE),
        "/utf-8",
        f"/I{py_include}",
        f"/I{pybind_inc}",
        "/link",
        f"/LIBPATH:{py_lib}",
        f"/OUT:{output}",
        f"/IMPLIB:{Path(__file__).parent / f'{MODULE}.lib'}",
        *EXTRA_LIBS,
    ]

    # 尝试直接调用 cl；没有就找 vcvars64.bat 借环境
    vcvars = None
    if not shutil.which("cl"):
        vcvars = find_vcvars()
        if not vcvars:
            raise BuildError(
                "未找到 MSVC 编译器（cl.exe）。\n"
                "    请安装 Visual Studio Build Tools，并勾选“使用 C++ 的桌面开发”。"
            )

    print("[*] 源文件：", SOURCE)
    print("[*] 输出：  ", output)
    print("[*] 命令：  ", " ".join(cl_args))
    print()

    if vcvars:
        # 在 vcvars 环境里执行 cl
        cmd = f'call "{vcvars}" >nul && ' + " ".join(f'"{a}"' if " " in a else a for a in cl_args)
        result = subprocess.run(cmd, shell=True)
    else:
        result = subprocess.run(cl_args)

    if result.returncode != 0:
        raise BuildError(f"编译失败，退出码 {result.returncode}")

    print(f"\n[✓] 编译成功：{output}")
    return output


if __name__ == "__main__":
    build()
