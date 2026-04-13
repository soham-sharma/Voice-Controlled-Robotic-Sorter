#!/usr/bin/env python3
import subprocess
import sys
import json
from vosk import Model, KaldiRecognizer

def main():
    print("Initializing Vosk... Downloading model if not present.")
    model = Model(lang="en-us")
    
    # Vosk prefers 16kHz for its standard models
    sample_rate = 16000
    rec = KaldiRecognizer(model, sample_rate)
    
    print("="*50)
    print("Microphone Test Started (Native PulseAudio). Please speak into the mic.")
    print("Press Ctrl+C to exit.")
    print("="*50)
    
    # Use parecord directly since we know it cleanly bridges to WSLg
    cmd = [
        "parecord", 
        "--rate=16000", 
        "--channels=1", 
        "--format=s16le", 
        "--raw",
        "-d", "@DEFAULT_SOURCE@"
    ]
    
    try:
        # Open parecord as a subprocess and read its stdout
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        
        while True:
            # Read chunks of raw audio data
            data = process.stdout.read(4000)
            if len(data) == 0:
                break
                
            if rec.AcceptWaveform(data):
                res = json.loads(rec.Result())
                text = res.get("text", "")
                if text:
                    print(f"Recognized: {text}")
            else:
                # Partial results can be printed here if needed
                pass
                
    except KeyboardInterrupt:
        print("\nStopping audio stream...")
        process.kill()
        print("Test finished.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()