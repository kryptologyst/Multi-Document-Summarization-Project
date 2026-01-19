#!/usr/bin/env python3
"""
Setup script for Multi-Document Summarization Project
"""

import os
import sys
import subprocess
import platform


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True


def install_dependencies():
    """Install required dependencies."""
    print("\n📦 Installing dependencies...")
    
    # Upgrade pip first
    if not run_command(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip"):
        return False
    
    # Install requirements
    if not run_command(f"{sys.executable} -m pip install -r requirements.txt", "Installing requirements"):
        return False
    
    return True


def download_models():
    """Download required models."""
    print("\n🤖 Downloading models...")
    
    if not run_command(f"{sys.executable} download_models.py", "Downloading models"):
        print("⚠️ Model download failed, but you can still use extractive summarization")
        return False
    
    return True


def create_directories():
    """Create necessary directories."""
    print("\n📁 Creating directories...")
    
    directories = [
        "data",
        "tests",
        "summarizer",
        ".streamlit"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    return True


def run_tests():
    """Run the test suite."""
    print("\n🧪 Running tests...")
    
    if not run_command(f"{sys.executable} -m pytest tests/ -v", "Running tests"):
        print("⚠️ Some tests failed, but the system should still work")
        return False
    
    return True


def main():
    """Main setup function."""
    print("🚀 Multi-Document Summarization Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    if not create_directories():
        print("❌ Failed to create directories")
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Download models
    download_models()  # Don't exit on failure, extractive summarization still works
    
    # Run tests
    run_tests()  # Don't exit on failure
    
    print("\n" + "=" * 50)
    print("✅ Setup completed successfully!")
    print("\n🎉 You can now use the system:")
    print("   • Web Interface: streamlit run app.py")
    print("   • Command Line: python main.py --help")
    print("   • Demo: python 0543.py")
    print("\n📚 For more information, see README.md")


if __name__ == "__main__":
    main()
