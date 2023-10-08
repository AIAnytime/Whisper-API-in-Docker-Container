from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from tempfile import NamedTemporaryFile
import whisper
import torch
from typing import List

# Checking if NVIDIA GPU is available
torch.cuda.is_available()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load the Whisper model:
model = whisper.load_model("base", device=DEVICE)

app = FastAPI()

@app.post("/whisper/")
async def handler(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files were provided")

    # For each file, let's store the results in a list of dictionaries.
    results = []

    for file in files:
        # Create a temporary file.
        with NamedTemporaryFile(delete=True) as temp:
            # Write the user's uploaded file to the temporary file.
            with open(temp.name, "wb") as temp_file:
                temp_file.write(file.file.read())
            
            # Let's get the transcript of the temporary file.
            result = model.transcribe(temp.name)

            # Now we can store the result object for this file.
            results.append({
                'filename': file.filename,
                'transcript': result['text'],
            })

    return JSONResponse(content={'results': results})


@app.get("/", response_class=RedirectResponse)
async def redirect_to_docs():
    return "/docs"
