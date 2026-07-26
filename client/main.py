"""
GPU Pod Desktop
===============
Entry point. Run with:  uv run python main.py
"""

from client.app import GPUApp

if __name__ == "__main__":
    app = GPUApp()
    app.run()
