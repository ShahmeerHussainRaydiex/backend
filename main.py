from pathlib import Path
import uvicorn
from fastapi import FastAPI, HTTPException, Header, UploadFile, File, Form
import requests
from pypexels import PyPexels
from helper import video_to_base64
import json
from dotenv import load_dotenv
import os

app = FastAPI()
load_dotenv()

client = ""
api_key = os.environ.get("OPENAI_API_KEY")

if api_key is not None:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


@app.get("/check")
async def root():
    import ffmpeg

    input_background = ffmpeg.input('background.mp4')
    input_color = ffmpeg.input('test.mp4')
    input_alpha = ffmpeg.input('test.mp4')

    coloralpha = ffmpeg.filter([input_color, input_alpha], 'alphamerge')

    background_resized = \
        input_background.filter('scale2ref', 'main_w*max(iw/main_w, ih/main_h)', 'main_h*max(iw/main_w, ih/main_h)')[0]
    overlay = ffmpeg.overlay(background_resized, coloralpha, shortest=1, x='main_w/2-overlay_w/2', y='main_h-overlay_h')

    darset = overlay.filter('setdar', 'dar=a').filter('pad', 'ceil(iw/2)*2', 'ceil(ih/2)*2')[0]
    output = darset

    output = output.output('output.mp4', vcodec='libx264', crf=17, pix_fmt='yuv420p', strict='experimental',
                           acodec='copy')

    output.run()

    output.run()

    return {f"message": f"Hello World"}


API_KEY = os.getenv("PEXEL")


@app.get("/search/images")
async def search_images(query: str):
    headers = {
        "Authorization": f"{API_KEY}"
    }

    # Make a request to the Pexels API
    response = requests.get(f"https://api.pexels.com/v1/search?query={query}", headers=headers)

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Pexels API request failed")
    response = response.json()
    image_url = response['photos'][0]['url']

    return image_url


def search_videos(queries):
    headers = {
        "Authorization": f"{API_KEY}"
    }
    urls = []
    # Make a request to the Pexels API
    for query in queries:
        try:
            response = requests.get(f"https://api.pexels.com/videos/search?query={query}&per_page=1",
                                headers=headers)
        except Exception as e:
                return {"error":e}

        if response.status_code != 200:
            continue
        response = response.json()
        link = response["videos"][0]["video_files"][0]["link"]
        urls.append(link)
    return urls


@app.get("/search/test")
async def search_video(query: str, num_page: int = 1, per_page: int = 1):
    headers = {
        "Authorization": API_KEY
    }
    py_pexel = PyPexels(api_key=API_KEY)
    search_videos_page = py_pexel.videos_search(query=query, per_page=per_page)
    for video in search_videos_page.entries:
        print(video.id, video.user.get('name'), video.url)

        url = f"https://api.pexels.com/videos/videos/{video.id}"
        response = requests.get(url, headers=headers)
        data_url = 'https://www.pexels.com/video/' + str(video.id) + '/download'
        r = requests.get(data_url)
        print(r.headers.get('content-type'))
        with open(f'{video.id}.mp4', 'wb') as outfile:
            outfile.write(r.content)
    return {f"message": f"Downloaded videos"}


@app.get("/video/video")
async def search_videos_stable(prompt: str):
    url = "https://modelslab.com/api/v6/video/video2video"
    video_path = "8625246.mp4"  # Provide the path to your video file
    base64_video = video_to_base64(video_path)
    print(base64_video)

    payload = json.dumps({
        "key": os.getenv("STABLE_DIFFUSION"),
        "model_id": "midjourney",
        "prompt": prompt,
        "negative_prompt": "low quality",
        "init_video": base64_video,
        "height": 512,
        "width": 512,
        "num_frames": 16,
        "num_inference_steps": 20,
        "guidance_scale": 7,
        "strength": 0.7,
        "base64": True,
        "webhook": None,
        "track_id": None
    })

    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)
    return response.json()


@app.post("/convert_text_to_speech/")
async def convert_text_to_speech(text: str):
    try:
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text
        )
        speech_file_path = Path(__file__).parent / "speech.mp3"
        response.stream_to_file(speech_file_path)
        return {"message": "Text converted to speech successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/transcribe_audio/")
async def transcribe_audio(audio_file: UploadFile = File(...)):
    try:
        with audio_file.file as file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=file
            )
        return {"transcription": transcription.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def generate_story(prompt: str="You're tasked with writing a story script "):
    try:
        story_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prompt}
            ],
            max_tokens=150
        )
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You will be provided with a block of text, and your task is to extract sentences of 3 to 4 words which can help to search videos on pexel."
                },
                {
                    "role": "user",
                    "content": story_response.choices[0].message.content
                }
            ],
            temperature=0.5,
            max_tokens=64,
            top_p=1
        )
        urls = search_videos(response.choices[0].message.content.split('\n'))
        return {"urls": urls,"keywords": response.choices[0].message.content ,"story": story_response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/regenerate_script")
async def regenerate_script(user_script: str, prompt_type: str):
    # Analyze user_script and determine which prompt to use
    # Based on prompt_type, select the appropriate prompt and generate script
    if prompt_type == "Shorter":
        prompt = "Make the script shorter:\n" + user_script
    elif prompt_type == "Longer":
        prompt = "Make the script longer:\n" + user_script
    elif prompt_type == "Casual":
        prompt = "Make the script sound casual:\n" + user_script
    elif prompt_type == "Professional":
        prompt = "Make the script sound professional:\n" + user_script
    else:
        return {"error": "Invalid prompt type"}

    # Use await to call the asynchronous function
    regenerated_script = generate_story(prompt)
    return {"regenerated_script": regenerated_script}


@app.post("/generate-story/")
async def story_generation(prompt: str):
    return {"response ": await generate_story(prompt)}


@app.get("/convert_text_to_music/")
async def convert_text_to_music(text: str):
    import requests

    url = "https://api.tryleap.ai/api/v1/music"

    payload = {
        "prompt": "An electronic music soundtrack with a trumpet solo",
        "mode": "melody",
        "duration": 28
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "X-Api-Key": "le_62f9c378_MH3yuPlrw4TCf5IRK57ANJGY"
    }

    response = requests.post(url, json=payload, headers=headers)

    print(response.text)
    return response.json()


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
