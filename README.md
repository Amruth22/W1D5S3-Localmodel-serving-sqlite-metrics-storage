# Gemini Proxy Server

Simple proxy server for Google's Gemini API with SQLite metrics storage.

## Features

- Local serving of Gemini AI models
- Metric storage in SQLite database
- Swagger UI documentation at `/docs`
- Health check endpoint at `/health`

## Setup

1. Clone this repository
2. Verify the `.env` file contains your Gemini API key
4. Install requirements:
   ```
   pip install -r requirements.txt
   ```
5. Run the server:
   ```
   python app.py
   ```

## API Usage

Send requests to `/gemini/generate` endpoint:

```json
{
  "prompt": {
    "text": "Tell me a joke about programming"
  },
  "generation_config": {
    "temperature": 0.7,
    "max_output_tokens": 100
  }
}
```

Response format:

```json
{
  "text": "Generated text response...",
  "metrics": {
    "prompt_tokens": 5,
    "completion_tokens": 20,
    "total_tokens": 25
  }
}
```

Visit `/docs` for complete API documentation.