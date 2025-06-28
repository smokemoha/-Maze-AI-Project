#!/usr/bin/env python3
"""
Quick demo script to run the maze AI project
"""

import os
import sys
import webbrowser
from pathlib import Path

def main():
    print("🤖 Maze AI Project Demo")
    print("=" * 40)
    
    # Get project directory
    project_dir = Path(__file__).parent
    
    print("\nAvailable demos:")
    print("1. Train the AI agent")
    print("2. Open web demo")
    print("3. View training results")
    print("4. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            print("\n🚀 Training AI agent...")
            os.system(f"cd {project_dir}/src && python train.py")
            
        elif choice == "2":
            print("\n🌐 Opening web demo...")
            web_demo_path = project_dir / "web_demo" / "index.html"
            webbrowser.open(f"file://{web_demo_path.absolute()}")
            print(f"Web demo opened at: file://{web_demo_path.absolute()}")
            
        elif choice == "3":
            print("\n📊 Viewing training results...")
            training_plot = project_dir / "training_progress.png"
            path_plot = project_dir / "maze_path.png"
            
            if training_plot.exists():
                print(f"Training progress plot: {training_plot}")
                os.system(f"xdg-open {training_plot} 2>/dev/null || open {training_plot} 2>/dev/null || echo 'Please open {training_plot} manually'")
            
            if path_plot.exists():
                print(f"Agent path visualization: {path_plot}")
                os.system(f"xdg-open {path_plot} 2>/dev/null || open {path_plot} 2>/dev/null || echo 'Please open {path_plot} manually'")
                
        elif choice == "4":
            print("\n👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please enter 1-4.")

if __name__ == "__main__":
    main()

