# Safety Filter Issues - Solutions Guide

## Problem

Medical images are sometimes blocked by Gemini API safety filters, even when safety settings are set to `BLOCK_NONE`. This is a known limitation of the API.

## What We've Implemented

### 1. Safety Settings Configuration
- All safety categories set to `BLOCK_NONE`
- Proper enum usage: `genai.types.HarmBlockThreshold.BLOCK_NONE`
- Medical context added to prompts

### 2. Better Error Handling
- Detailed safety rating logging
- Clear error messages with blocked categories
- Suggestions for resolution

## Solutions & Workarounds

### Solution 1: Use Different Images
Some images trigger filters more than others:
- Try using a different medical image
- Images with less anatomical detail may work better
- Cropped/anonymized images sometimes work better

### Solution 2: Contact Google Cloud Support
For medical use cases, you can:
- Request API exemptions for medical imaging
- Apply for Vertex AI which may have different policies
- Get whitelisted for medical research use cases

### Solution 3: Pre-process Images
Reduce triggers by:
- Cropping to focus on specific anatomical regions
- Adding medical image metadata/headers
- Converting to different formats
- Adding medical image annotations

### Solution 4: Use Vertex AI API
Vertex AI may have different safety policies:
```python
# Alternative: Use Vertex AI instead of Gemini API
# Contact Google Cloud for Vertex AI access for medical use
```

### Solution 5: Modify Prompts
Add more explicit medical context:
```python
prompt = "This is a clinical diagnostic image for medical analysis. Analyze for medical conditions only."
```

## Current Configuration

```python
safety_settings = [
    {
        "category": genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE
    },
    {
        "category": genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE
    },
    {
        "category": genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE
    },
    {
        "category": genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE
    }
]
```

## Testing Safety Filter Handling

To test if safety filters are working:

```bash
# Run the diagnostic
python scripts/diagnose_api.py

# Test with an image
python scripts/test_disease_detection.py /path/to/image.jpg
```

## Known Limitations

1. **API-Level Limitation**: Even with `BLOCK_NONE`, some medical images may trigger filters
2. **No Guaranteed Workaround**: There's no 100% reliable workaround at the API level
3. **Image-Dependent**: Some images work, others don't - it's inconsistent

## Recommended Actions

1. **For Development/Testing**:
   - Try multiple different medical images
   - Use anonymized or de-identified images when possible
   - Keep detailed logs of which images work/fail

2. **For Production**:
   - Contact Google Cloud Support for medical use case exemptions
   - Consider using Vertex AI for production medical imaging
   - Implement fallback mechanisms for blocked images

3. **For Research**:
   - Document which image types trigger filters
   - Request research access/whitelisting
   - Use approved medical image datasets

## Error Message Interpretation

When you see:
```
Response blocked by safety filters (finish_reason: SAFETY).
Blocked by: HARM_CATEGORY_SEXUALLY_EXPLICIT (HIGH)
```

This means:
- The image triggered the sexually explicit category
- Even though we set `BLOCK_NONE`, the API still blocked it
- This is an API-level limitation, not a code issue

## Next Steps

1. ✅ Code updated with proper safety settings
2. ✅ Better error messages implemented
3. ⚠️ Still encountering blocks? Try:
   - Different medical images
   - Contacting Google Cloud Support
   - Using Vertex AI API

---

**Note**: This is a known issue with Gemini API and medical images. The code is configured correctly, but API-level limitations may still cause blocks.



