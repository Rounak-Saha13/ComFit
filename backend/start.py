"""
Startup script for Comfit Copilot Backend, includes checking/creating .env file and installing dependencies
"""

import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def check_environment():
    """Check if required environment variables are set"""
    required_vars = [
        "DATABASE_URL",
        "SUPABASE_URL", 
        "SUPABASE_ANON_KEY"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease create a .env file with the required variables.")
        print("See README.md for setup instructions.")
        return False
    
    print("✅ Environment variables configured")
    return True

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import fastapi
        import sqlalchemy
        import supabase
        import uvicorn
        print("✅ Dependencies installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def create_env_template():
    """Create .env template if it doesn't exist"""
    env_file = Path(".env")
    if not env_file.exists():
        print("📝 Creating .env template...")
        template = """# Database Configuration
DATABASE_URL=postgresql://postgres:password@localhost:5432/comfit

# Supabase Configuration
SUPABASE_URL=your_supabase_url_here
SUPABASE_ANON_KEY=your_supabase_anon_key_here

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434

# Server Configuration
PORT=8000

# Google API Keys (for web search functionality)
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_CSE_ID=your_google_cse_id_here
"""
        with open(env_file, "w") as f:
            f.write(template)
        print("✅ Created .env template. Please update it with your actual values.")
        return False
    return True

def main():
    print("🚀 Starting Comfit Copilot Backend...")
    print("=" * 50)
    
    # Check if .env exists
    if not create_env_template():
        return
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Check environment variables
    if not check_environment():
        return
    
    print("\n🎯 Starting server...")
    print("📖 API documentation will be available at: http://localhost:8000/docs")
    print("🔗 Health check: http://localhost:8000/health")
    print("=" * 50)
    
    # Start the server
    try:
        import uvicorn
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=int(os.getenv("PORT", 8000)),
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

if __name__ == "__main__":
    main() 