#!/usr/bin/env python3
"""
Setup script for Movie Recommendation System
This script helps you set up the project environment and data.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during {description}")
        print(f"Command: {command}")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 7:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible!")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor} is not compatible. Please use Python 3.7+")
        return False

def install_dependencies():
    """Install required Python packages."""
    packages = [
        "streamlit==1.29.0",
        "pandas==2.0.3", 
        "numpy==1.24.3",
        "scikit-learn==1.3.0",
        "matplotlib==3.7.2",
        "seaborn==0.12.2",
        "plotly==5.17.0",
        "requests==2.31.0",
        "scipy==1.11.1"
    ]
    
    print("📦 Installing Python packages...")
    for package in packages:
        success = run_command(f"pip install {package}", f"Installing {package.split('==')[0]}")
        if not success:
            print(f"⚠️  Warning: Failed to install {package}")
    
    return True

def prepare_project_structure():
    """Ensure project directory structure is correct."""
    print("📁 Setting up project structure...")
    
    # Create necessary directories
    directories = ["data", "models", "temp"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"📂 Created/verified directory: {directory}")
    
    # Check if all required files exist
    required_files = [
        "app.py",
        "recommendation_engine.py", 
        "prepare_data.py",
        "requirements.txt",
        "README.md"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing files: {', '.join(missing_files)}")
        return False
    else:
        print("✅ All required files are present!")
        return True

def main():
    """Main setup function."""
    print("🎬 Movie Recommendation System Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Prepare project structure
    if not prepare_project_structure():
        print("❌ Project structure setup failed!")
        sys.exit(1)
    
    # Install dependencies
    print("\n📦 Installing dependencies...")
    install_success = run_command("pip install -r requirements.txt", "Installing requirements")
    
    if not install_success:
        print("⚠️  Requirements installation had issues. Trying individual packages...")
        install_dependencies()
    
    # Prepare dataset
    print("\n📊 Preparing dataset...")
    data_success = run_command("python prepare_data.py", "Preparing MovieLens dataset")
    
    if not data_success:
        print("⚠️  Dataset preparation failed. You can run 'python prepare_data.py' manually later.")
    
    # Final instructions
    print("\n🎉 Setup Complete!")
    print("=" * 50)
    print("📋 Next Steps:")
    print("1. Navigate to the project directory:")
    print("   cd C:\\Users\\FAIJ\\movie-recommendation-system")
    print()
    print("2. If dataset preparation failed, run:")
    print("   python prepare_data.py")
    print()
    print("3. Start the application:")
    print("   streamlit run app.py")
    print()
    print("4. Open your browser and go to:")
    print("   http://localhost:8501")
    print()
    print("🎬 Happy movie recommending!")

if __name__ == "__main__":
    main()
