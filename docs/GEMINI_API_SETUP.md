# Gemini API Setup Guide

## Quick Setup

### Step 1: Get Your API Key

1. Go to: **https://aistudio.google.com/app/apikey**
2. Sign in with your Google account
3. Click **"Create API Key"** or use an existing key
4. Copy the API key (it will look like: `AIzaSy...`)

### Step 2: Set Up the API Key

**Option A: Using the Setup Script (Recommended)**
```bash
source .venv/bin/activate
python scripts/setup_gemini_api_key.py
```

The script will:
- Prompt you to enter your API key (hidden input for security)
- Save it to the `.env` file
- Verify the key format

**Option B: Manual Setup**
1. Create/edit `.env` file in the project root:
```bash
nano .env
```

2. Add this line (replace with your actual API key):
```
GOOGLE_API_KEY=AIzaSyYourActualAPIKeyHere
```

3. Save and exit (Ctrl+X, then Y, then Enter in nano)

### Step 3: Verify Setup

Run the diagnostic script:
```bash
python scripts/test_gemini_setup.py
```

This will:
- ✅ Check if the API key is set correctly
- ✅ Test the API connection
- ✅ List available models
- ✅ Verify everything works

## Troubleshooting

### Error: "API key appears to be invalid"
- Make sure the API key is 30-40 characters long
- Should start with `AIza`
- No extra spaces or quotes around it
- Run `python scripts/setup_gemini_api_key.py` again

### Error: "API connection failed"
- Check your internet connection
- Verify the API key is correct at https://aistudio.google.com/app/apikey
- Check if your API key has quota/usage limits
- Make sure you have billing enabled (if required)

### Error: "GOOGLE_API_KEY not found"
- Make sure `.env` file exists in project root
- Check the file contains: `GOOGLE_API_KEY=your-key-here`
- Make sure you're running scripts from the project root
- Try: `source .venv/bin/activate` to ensure environment is loaded

## Testing the Setup

Once setup is complete, test with:
```bash
# Test RAG system
python scripts/test_rag_with_record.py --test 130656 --questions "What are my diagnoses?"

# Or test simple inference
python src/training/gemini_inference.py --input "Test text"
```

## Security Notes

- ✅ The `.env` file is in `.gitignore` - it won't be committed to git
- ✅ Never share your API key publicly
- ✅ Regenerate your API key if it's been exposed
- ✅ Use environment variables for production deployments


