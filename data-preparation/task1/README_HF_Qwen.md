# Hugging Face Qwen Integration for CityLLaVA

This document explains how to use the updated `shortQA_split.py` with Hugging Face Inference API for Qwen models.

## Changes Made

1. **Original file preserved**: The original `shortQA_split.py` has been renamed to `shortQA_split_original.py`
2. **New implementation**: A new `shortQA_split.py` file now supports Hugging Face Inference API
3. **Backward compatibility**: The original Dashscope and OpenAI implementations are still available

## Setup

### 1. Get Hugging Face API Token

1. Go to [Hugging Face Settings](https://huggingface.co/settings/tokens)
2. Create a new token with "Read" permissions
3. Copy the token (starts with `hf_`)

### 2. Install Required Dependencies

```bash
pip install requests
```

## Usage

### Using Hugging Face Qwen (Recommended)

```bash
python shortQA_split.py --model HF_Qwen --api-key YOUR_HF_TOKEN
```

### Using Original Dashscope Qwen

```bash
python shortQA_split.py --model Qwen --api-key YOUR_DASHSCOPE_KEY
```

### Using OpenAI GPT-4

```bash
python shortQA_split.py --model Openai --api-key YOUR_OPENAI_KEY
```

## Available Qwen Models

The default model is `Qwen/Qwen2.5-72B-Instruct`. You can modify the `model_id` variable in the code to use other Qwen models:

- `Qwen/Qwen2-72B-Instruct`
- `Qwen/Qwen2.5-72B-Instruct`
- `Qwen/Qwen2-7B-Instruct`
- `Qwen/Qwen2.5-7B-Instruct`
- And other Qwen models available on Hugging Face

## Configuration

### Model Parameters

The Hugging Face implementation uses these parameters:

```python
"parameters": {
    "max_new_tokens": 512,
    "temperature": 0.1,
    "top_p": 0.9,
    "do_sample": True,
    "return_full_text": False
}
```

### Error Handling

- **Automatic retries**: Up to 3 attempts with exponential backoff
- **Model loading**: Handles 503 errors when models are loading
- **Timeout**: 60-second timeout per request

## Advantages of Hugging Face Inference API

1. **No local setup**: No need to download and host large models
2. **Cost-effective**: Pay-per-use pricing
3. **Multiple models**: Easy access to various Qwen model sizes
4. **Scalability**: Handles traffic spikes automatically
5. **Latest models**: Access to newest Qwen releases

## Troubleshooting

### Common Issues

1. **Model loading (503 error)**: The script automatically waits and retries
2. **Rate limiting**: Reduce the number of parallel processes if needed
3. **Token issues**: Ensure your HF token has the correct permissions

### Performance Tips

1. **Model choice**: Use smaller models (7B) for faster responses if quality is acceptable
2. **Batch processing**: The script already uses multiprocessing for efficiency
3. **Retry logic**: The built-in retry mechanism handles temporary failures

## Migration from Original Implementation

To migrate from the original Dashscope implementation:

1. Get a Hugging Face API token
2. Change the model parameter: `--model HF_Qwen`
3. Update the API key parameter: `--api-key YOUR_HF_TOKEN`

The output format and functionality remain exactly the same.

## Example Usage in Training Pipeline

Update your `prepare_data_train.sh` script:

```bash
# Original (Dashscope)
# python shortQA_split.py --model Qwen --api-key $DASHSCOPE_KEY

# New (Hugging Face)
python shortQA_split.py --model HF_Qwen --api-key $HF_TOKEN
```

## Support

If you encounter issues:

1. Check the Hugging Face model status page
2. Verify your API token permissions
3. Review the console output for specific error messages
4. Fall back to the original implementation if needed: `shortQA_split_original.py`