import subprocess

scripts = ["core/benchmark_procedural_noOSS.py", "core/OSS_solver.py", "core/Vectorized_OSS_solver.py"]
for script in scripts:
    subprocess.run(["python", script], check=True)