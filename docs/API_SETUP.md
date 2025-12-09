# API Setup Guide

## Gemini API Setup

### Step 1: Get Your API Key

1. Go to: **https://aistudio.google.com/app/apikey**
2. Sign in with your Google account
3. Click **"Create API Key"** or use an existing key
4. Copy the API key (it will look like: `AIzaSy...`)

### Step 2: Set Up the API Key

**Option A: Using the Setup Script (Recommended)**
```bash
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
- Check for extra spaces or quotes
- Run `python scripts/setup_gemini_api_key.py` again

### Error: "GOOGLE_API_KEY not found"
- Check the `.env` file contains: `GOOGLE_API_KEY=your-key-here`
- Make sure `.env` is in the project root directory
- Restart your terminal/IDE after creating `.env`

### Error: API Access Issues

If you're being redirected when trying to access the API key page, see [Troubleshooting Guide](TROUBLESHOOTING.md#api-key-access-issues).

## Security Best Practices

- ✅ Never commit your API key to version control
- ✅ Use `.env` file (already in `.gitignore`)
- ✅ Regenerate your API key if it's been exposed
- ✅ Use environment variables in production

## Google Cloud Setup

For advanced features, you may need Google Cloud credentials. See [GCP Setup Guide](GCP_SETUP.md).
