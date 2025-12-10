# Troubleshooting: Gemini API Key Access Issues

## Problem: Getting Redirected to "Available Regions" Page

If you're being redirected when trying to access https://aistudio.google.com/app/apikey, try these solutions:

### Solution 1: Check Account Requirements
- ✅ Make sure you're signed in with a Google account
- ✅ Verify you're 18+ (required for Google AI Studio)
- ✅ Try a different Google account if available
- ✅ Ensure your Google account is in good standing

### Solution 2: Clear Browser Data
1. Clear cookies and cache for `aistudio.google.com`
2. Try an incognito/private window
3. Disable browser extensions temporarily
4. Try a different browser (Chrome, Firefox, Safari)

### Solution 3: Direct URL Access
Try these direct links:
- https://aistudio.google.com/app/apikey
- https://makersuite.google.com/app/apikey (older URL)
- https://console.cloud.google.com/apis/credentials (Google Cloud Console)

### Solution 4: Use Google Cloud Console (Alternative)
If Google AI Studio doesn't work, you can create API keys through Google Cloud Console:

1. Go to: https://console.cloud.google.com/
2. Create a new project (or use existing)
3. Enable the "Generative Language API"
4. Go to: APIs & Services → Credentials
5. Create API Key
6. Restrict the key to "Generative Language API" for security

### Solution 5: Check Network/VPN
- If using a VPN, try disabling it
- If behind a corporate firewall, you may need network admin help
- Try from a different network (mobile hotspot, etc.)

### Solution 6: Use Vertex AI (Enterprise Alternative)
If Google AI Studio isn't accessible, you can use Vertex AI:
- More complex setup, requires Google Cloud account
- Better for enterprise/production use
- See: https://cloud.google.com/vertex-ai/docs

### Solution 7: Alternative - Use Existing Key
If you have an existing API key that might still work, you can try to restore it:
1. Check if you have it saved anywhere (password manager, notes, etc.)
2. If you find an old key, test it using the test script below
3. **Important**: Never commit API keys to version control - always use environment variables or `.env` files (which are in `.gitignore`)

## Quick Test

After getting your API key (by any method above), test it:

```bash
# Set the key
export GOOGLE_API_KEY="your-key-here"

# Or add to .env file
echo "GOOGLE_API_KEY=your-key-here" >> .env

# Test it
python scripts/test_gemini_setup.py
```

## Still Having Issues?

1. **Check Google AI Studio Status**: https://status.cloud.google.com/
2. **Try from different device**: Phone, tablet, or another computer
3. **Contact Google Support**: If account-related issues persist
4. **Check Google Account Settings**: Ensure AI features are enabled


