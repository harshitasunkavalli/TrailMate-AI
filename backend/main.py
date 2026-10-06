import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from huggingface_hub import InferenceClient

# Load environment variables from .env
load_dotenv()

# Create FastAPI application
app = FastAPI(
    title="TrailMate AI",
    description=(
        "An AI-powered outdoor companion that helps people "
        "spend less time on screens and more time outdoors."
    ),
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get Hugging Face token
HF_TOKEN = os.getenv("HF_TOKEN")

# Create Hugging Face inference client
client = InferenceClient(
    provider="featherless-ai",
    api_key=HF_TOKEN
)


# Request model
class ChallengeRequest(BaseModel):
    duration: int
    mood: str
    activity_type: str


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Welcome to TrailMate AI 🌿",
        "status": "running"
    }


# Challenge endpoint
@app.post("/challenge")
def create_challenge(request: ChallengeRequest):

    prompt = f"""
You are TrailMate AI.

Create a short outdoor challenge for a person who wants to spend
less time on their phone.

Time available: {request.duration} minutes
Mood: {request.mood}
Activity: {request.activity_type}

Give exactly:
TITLE: one short title
MISSION: one short sentence
STEP 1: one simple outdoor action
STEP 2: one simple outdoor action
STEP 3: one simple outdoor action
TIP: one short motivating sentence

Rules:
- All activities must fit within {request.duration} minutes.
- No money or equipment.
- No cycling, swimming, fishing, climbing or driving.
- Use simple activities such as walking, observing trees,
  listening to birds, noticing clouds, breathing or mindfulness.
- Keep everything safe and beginner-friendly.
- Do not write anything else.
"""



    # Ask the open-weight model to generate the challenge
    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=250,
        temperature=0.7
    )

    # Extract generated text
    generated_text = response.choices[0].message.content

    # Return the generated challenge
    return {
        "duration": request.duration,
        "mood": request.mood,
        "activity_type": request.activity_type,
        "challenge": generated_text
    }