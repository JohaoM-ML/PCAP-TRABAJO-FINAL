#!/usr/bin/env python3
"""
Macroeconomic Dashboard Launcher
Run this script to launch the interactive macroeconomic dashboard
"""

import subprocess
import sys
import os

def main():
    """Launch the Streamlit dashboard"""
    print("🚀 Launching Macroeconomic Dashboard...")
    print("📊 The dashboard will open in your default web browser")
    print("🔧 Use Ctrl+C to stop the dashboard when you're done")
    print("-" * 50)
    
    try:
        # Run the Streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "macroeconomic_dashboard.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped. Thank you for using the Macroeconomic Dashboard!")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")
        print("💡 Make sure you have installed all requirements: pip install -r requirements.txt")

if __name__ == "__main__":
    main()

