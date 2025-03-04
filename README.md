from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import requests

# Load API keys from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
runway_api_key = os.getenv("RUNWAY_API_KEY")

elevenlabs_url = "https://api.elevenlabs.io/v1/text-to-speech"
runway_url = "https://api.runwayml.com/v1/generate-video"

app = FastAPI()

# Enable CORS for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class VideoRequest(BaseModel):
    topic: str
    category: str

@app.post("/api/generate-video")
async def generate_video(request: VideoRequest):
    try:
        # Generate script using OpenAI GPT
        prompt = f"Generate a viral video script about {request.topic} in the {request.category} category."
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        script = response["choices"][0]["message"]["content"]
        
        # Convert script to speech using ElevenLabs
        tts_response = requests.post(
            f"{elevenlabs_url}/default",
            headers={
                "xi-api-key": elevenlabs_api_key,
                "Content-Type": "application/json"
            },
            json={"text": script, "voice_settings": {"stability": 0.5, "similarity_boost": 0.8}}
        )
        
        if tts_response.status_code != 200:
            raise HTTPException(status_code=500, detail="TTS conversion failed")
        
        audio_url = tts_response.json().get("audio_url", "")
        
        # Generate AI Video using RunwayML
        video_response = requests.post(
            runway_url,
            headers={
                "Authorization": f"Bearer {runway_api_key}",
                "Content-Type": "application/json"
            },
            json={"text": script, "voiceover": audio_url}
        )
        
        if video_response.status_code != 200:
            raise HTTPException(status_code=500, detail="Video generation failed")
        
        video_url = video_response.json().get("video_url", "")
        
        return {"script": script, "audioUrl": audio_url, "videoUrl": video_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Dockerfile for deployment
DOCKERFILE_CONTENT = """
# Use official Python image
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose the application port
EXPOSE 8000

# Run the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
"""

# Generate Dockerfile for deployment
with open("Dockerfile", "w") as f:
    f.write(DOCKERFILE_CONTENT)
