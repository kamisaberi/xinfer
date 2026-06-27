#!/usr/bin/env python3
import os
import subprocess
import sys
import shutil
import platform

# --- UI Colors ---
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_status(name, status, message=""):
    if status == "PASS":
        print(f"[{Colors.GREEN}PASS{Colors.ENDC}] {name}")
    elif status == "WARN":
        print(f"[{Colors.YELLOW}WARN{Colors.ENDC}] {name} - {message}")
    elif status == "FAIL":
        print(f"[{Colors.RED}FAIL{Colors.ENDC}] {name} - {message}")

def check_cmd(cmd):
    return shutil.which(cmd) is not None

def check_py_pkg(pkg):
    try:
        subprocess.check_output([sys.executable, "-c", f"import {pkg}"], stderr=subprocess.STDOUT)
        return True
    except:
        return False

def check_docker_image(image_name):
    if not check_cmd("docker"): return False
    try:
        output = subprocess.check_output(["docker", "images", "-q", image_name]).decode().strip()
        return len(output) > 0
    except:
        return False

def run_doctor():
    print(f"{Colors.HEADER}{Colors.BOLD}xInfer Multi-Platform Diagnostic Tool (v1.0){Colors.ENDC}\n")
    print(f"Operating System: {platform.system()} {platform.release()}")
    print(f"Python Version: {sys.version.split()[0]}\n")

    # --- CORE TOOLS ---
    print(f"{Colors.BLUE}--- Core Build Tools ---{Colors.ENDC}")
    print_status("CMake", "PASS" if check_cmd("cmake") else "FAIL", "Install via 'sudo apt install cmake'")
    print_status("Docker", "PASS" if check_cmd("docker") else "WARN", "Required for Vitis-AI and specialized toolchains")
    print_status("Protobuf", "PASS" if check_cmd("protoc") else "WARN", "Required for ONNX model surgery")

    # --- 1. DESKTOP/SERVER ---
    print(f"\n{Colors.BLUE}--- Desktop & Server Backends ---{Colors.ENDC}")
    print_status("NVIDIA TensorRT", "PASS" if check_cmd("trtexec") else "FAIL", "TRT not in PATH or not installed.")
    print_status("Intel OpenVINO", "PASS" if check_py_pkg("openvino") else "FAIL", "Run 'pip install openvino-dev'")
    print_status("AMD Ryzen AI", "PASS" if check_py_pkg("voe") else "WARN", "Requires Ryzen AI Software (VOE).")
    if platform.system() == "Darwin":
        print_status("Apple CoreML", "PASS" if check_cmd("xcrun") else "FAIL")
    else:
        print_status("Apple CoreML", "WARN", "Can only compile .mlmodelc on macOS.")

    # --- 2. FPGA & DEFENSE (AEGIS SKY) ---
    print(f"\n{Colors.BLUE}--- FPGA & Adaptive (Aegis Sky Tier) ---{Colors.ENDC}")
    vitis_img = "xilinx/vitis-ai-cpu"
    print_status("AMD Vitis-AI", "PASS" if check_docker_image(vitis_img) else "FAIL", f"Missing docker image '{vitis_img}'")
    print_status("Intel FPGA AI Suite", "WARN", "Requires Quartus Prime and Manual Plugin check.")
    print_status("Microchip VectorBlox", "PASS" if os.getenv("VECTORBLOX_SDK") else "WARN", "Set VECTORBLOX_SDK env var.")
    print_status("Lattice sensAI", "WARN", "Lattice Diamond/sensAI is proprietary (license required).")

    # --- 3. MOBILE & SOC ---
    print(f"\n{Colors.BLUE}--- Mobile & Mobile-NPU ---{Colors.ENDC}")
    print_status("Qualcomm QNN", "PASS" if os.getenv("QNN_SDK_ROOT") else "FAIL", "Set QNN_SDK_ROOT to your Qualcomm SDK path.")
    print_status("Rockchip RKNN", "PASS" if check_py_pkg("rknn.api") else "FAIL", "Run 'pip install rknn-toolkit2'")
    print_status("MediaTek NeuroPilot", "WARN", "NeuroPilot SDK usually requires manual installation.")
    print_status("Samsung Exynos", "WARN", "Exynos AI Studio check not implemented.")

    # --- 4. SPECIALIZED EDGE (BLACKBOX SIEM) ---
    print(f"\n{Colors.BLUE}--- Specialized Edge (Blackbox SIEM Tier) ---{Colors.ENDC}")
    print_status("Hailo Dataflow", "PASS" if check_py_pkg("hailo_sdk_client") else "WARN", "Run 'pip install hailo_sdk_client'")
    print_status("Ambarella CVFlow", "WARN", "Requires NDA access to Ambarella Cooper Portal.")
    print_status("Google Edge TPU", "PASS" if check_cmd("edgetpu_compiler") else "WARN", "Install via 'apt-get install edgetpu-compiler'")

    print(f"\n{Colors.HEADER}{Colors.BOLD}Diagnostic Complete.{Colors.ENDC}")
    print("For Aegis Sky: Ensure Vitis-AI and QNN are PASS.")
    print("For Blackbox SIEM: Ensure Rockchip and OpenVINO are PASS.")

if __name__ == "__main__":
    run_doctor()