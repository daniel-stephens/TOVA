import subprocess

# Start FastAPI with uvicorn
uvicorn_cmd = [
    "python", "-m", "uvicorn", "src.api.main:app",
    "--reload", "--port", "8989"
]

# Start Flask (assumes FLASK_APP is set, or uses app.py by default)
flask_cmd = ["flask", "run"]

# Launch both processes
uvicorn_proc = subprocess.Popen(uvicorn_cmd)
flask_proc = subprocess.Popen(flask_cmd)

# Optional: wait for both to exit (Ctrl+C stops them together)
uvicorn_proc.wait()
flask_proc.wait()
