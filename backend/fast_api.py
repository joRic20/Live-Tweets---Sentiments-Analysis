from fastapi import FastAPI
from pydantic import BaseModel
import subprocess
import signal

app = FastAPI()

# Keep reference to the current ingest process so we can stop/start as needed
current_process = None

# Define the shape of the incoming POST request (expects a JSON with a "term" string)
class StreamRequest(BaseModel):
    term: str

@app.post("/start_stream")
def start_stream(req: StreamRequest):
    """
    Endpoint to trigger or update the ingest.py process with a new search term.
    """
    global current_process
    # If there's an existing ingest process, kill it
    if current_process is not None:
        current_process.send_signal(signal.SIGTERM)
    # Start ingest.py with the user-supplied term (runs as new process)
    current_process = subprocess.Popen(["python", "ingest.py", req.term])
    # Return confirmation
    return {"status": "started", "term": req.term}
