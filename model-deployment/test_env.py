# test_env.py in model-deployment directory
import os
from pathlib import Path
from dotenv import load_dotenv

print("="*50)
print("ENVIRONMENT VARIABLE DEBUG")
print("="*50)

# Method 1: Direct load
load_dotenv('.env')
key1 = os.getenv("GEMINI_API_KEY")
print(f"Method 1 - Direct: {key1[:10] if key1 else 'NOT FOUND'}")

# Method 2: Path object
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)
key2 = os.getenv("GEMINI_API_KEY")
print(f"Method 2 - Path: {key2[:10] if key2 else 'NOT FOUND'}")

# Method 3: Absolute path
abs_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=abs_path, override=True)
key3 = os.getenv("GEMINI_API_KEY")
print(f"Method 3 - Absolute: {key3[:10] if key3 else 'NOT FOUND'}")

# Show what's actually in environment
print(f"\nActual env value: {os.environ.get('GEMINI_API_KEY', 'NOT IN ENVIRONMENT')[:10]}")

# Check if file exists
print(f"\n.env file exists: {Path('.env').exists()}")
print(f"Current directory: {os.getcwd()}")