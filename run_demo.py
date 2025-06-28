"""
Quick demo script to run the maze AI project
"""

# Import required libraries
import os  # For executing system commands and interacting with the operating system
import sys  # For system-specific parameters and functions
import webbrowser  # For opening the web demo in a browser
from pathlib import Path  # For handling file paths in a platform-independent way

def main():
    # Print a welcome message and a separator line for clarity
    print("🤖 Maze AI Project Demo")
    print("=" * 40)
    
    # Get the directory of the current script using Path
    # Path(__file__).parent resolves to the directory containing this script
    project_dir = Path(__file__).parent
    
    # Display the available options for the user
    print("\nAvailable demos:")
    print("1. Train the AI agent")
    print("2. Open web demo")
    print("3. View training results")
    print("4. Exit")
    
    # Start an infinite loop to keep prompting the user until they choose to exit
    while True:
        # Get user input and remove leading/trailing whitespace
        choice = input("\nEnter your choice (1-4): ").strip()
        
        # Option 1: Train the AI agent
        if choice == "1":
            print("\n🚀 Training AI agent...")
            # Navigate to the src directory and run train.py using a system command
            # The f-string constructs the command: `cd {project_dir}/src && python train.py`
            os.system(f"cd {project_dir}/src && python train.py")
            
        # Option 2: Open the web-based demo
        elif choice == "2":
            print("\n🌐 Opening web demo...")
            # Construct the path to the index.html file in the web_demo directory
            web_demo_path = project_dir / "web_demo" / "index.html"
            # Open the HTML file in the default web browser using a file:// URL
            webbrowser.open(f"file://{web_demo_path.absolute()}")
            # Inform the user of the file path for reference
            print(f"Web demo opened at: file://{web_demo_path.absolute()}")
            
        # Option 3: View training results (plots)
        elif choice == "3":
            print("\n📊 Viewing training results...")
            # Define paths to the training progress and maze path visualization images
            training_plot = project_dir / "training_progress.png"
            path_plot = project_dir / "maze_path.png"
            
            # Check if the training progress plot exists before attempting to open it
            if training_plot.exists():
                print(f"Training progress plot: {training_plot}")
                # Attempt to open the image using a system command
                # xdg-open is used on Linux, open on macOS; fallback message if neither works
                os.system(f"xdg-open {training_plot} 2>/dev/null || open {training_plot} 2>/dev/null || echo 'Please open {training_plot} manually'")
            
            # Check if the maze path visualization exists before attempting to open it
            if path_plot.exists():
                print(f"Agent path visualization: {path_plot}")
                # Same logic as above to open the image, with a fallback message
                os.system(f"xdg-open {path_plot} 2>/dev/null || open {path_plot} 2>/dev/null || echo 'Please open {path_plot} manually'")
                
        # Option 4: Exit the program
        elif choice == "4":
            print("\n👋 Goodbye!")
            # Break the loop to exit the program
            break
            
        # Handle invalid input
        else:
            print("❌ Invalid choice. Please enter 1-4.")

# Standard Python idiom to run the main function when the script is executed directly
if __name__ == "__main__":
    main()