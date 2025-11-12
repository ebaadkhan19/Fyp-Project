import os
import io
import collections
from google.cloud import speech
from google.cloud import texttospeech
from groq import Groq
import pyaudio  # For microphone input
import pygame  # For audio output
import webrtcvad  # For Voice Activity Detection

# --- 1. API Client Initialization ---

# Check for API keys
if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
    print("Error: GOOGLE_APPLICATION_CREDENTIALS not set.")
    print(
        r"Please run: $env:GOOGLE_APPLICATION_CREDENTIALS = 'C:\path\to\your\credentials.json'"
    )
    exit()
if "GROQ_API_KEY" not in os.environ:
    print("Error: GROQ_API_KEY not set.")
    print("Please run: $env:GROQ_API_KEY = 'your_groq_key_here'")
    exit()

# Initialize all clients
try:
    groq_client = Groq()
    tts_client = texttospeech.TextToSpeechClient()
    speech_client = speech.SpeechClient()
    print("API clients initialized.")
except Exception as e:
    print(f"Error initializing clients: {e}")
    exit()

# Initialize Pygame Mixer for audio playback
try:
    pygame.mixer.init()
    print("Audio player initialized.")
except Exception as e:
    print(f"Error initializing audio player: {e}")
    exit()

# --- 2. VAD & Audio Configuration ---

# VAD needs specific frame durations (10, 20, or 30 ms)
FRAME_DURATION_MS = 30
RATE = 16000
CHUNK = int(RATE * FRAME_DURATION_MS / 1000)  # 480 samples
VAD_AGGRESSIVENESS = 3  # 0 to 3 (3 is most aggressive at filtering non-speech)

# Language and Voice Settings
STT_LANGUAGE = "ur-PK"
TTS_LANGUAGE_CODE = "ur-PK"

# --- 3. Helper Functions (The 3 Services) ---


def call_groq(text):
    """Sends text to Groq and returns the response."""
    print(f"User: {text}")
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. You must respond ONLY in Urdu.",
                },
                {"role": "user", "content": text},
            ],
            model="llama-3.1-8b-instant",
            temperature=0.7,
        )
        response_text = chat_completion.choices[0].message.content
        print(f"Bot: {response_text}")
        return response_text
    except Exception as e:
        print(f"Error calling Groq: {e}")
        return "معاف کیجئے گا، مجھے جواب سوچنے میں دشواری ہوئی۔"


def call_tts(text):
    """Sends text to Google TTS and returns the audio content."""
    try:
        synthesis_input = texttospeech.SynthesisInput(text=text)

        # --- THIS IS THE FIX ---
        # We let Google pick the default voice by NOT specifying a 'name'.
        voice = texttospeech.VoiceSelectionParams(
            language_code=TTS_LANGUAGE_CODE,
            ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL,
        )

        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )

        response = tts_client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
        return response.audio_content
    except Exception as e:
        print(f"Error calling TTS: {e}")
        return None


def play_audio(audio_content):
    """Plays audio content from memory using Pygame."""
    if not audio_content:
        return
    try:
        audio_file = io.BytesIO(audio_content)
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        print("Bot finished speaking.")
    except Exception as e:
        print(f"Error playing audio: {e}")


# --- 4. The New VAD-Based Main Loop ---


def main_loop():
    """Listens for speech with VAD and processes it."""

    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    p = pyaudio.PyAudio()

    print("\n--- Starting Conversation (Speak into your mic) ---")

    try:
        while True:
            # Open a new stream for this turn
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
            )

            print("\nListening...")

            # --- This is the VAD loop ---
            frames = collections.deque()
            speaking = False
            silent_frames_count = 0
            # We want ~0.75 seconds of silence to trigger end-of-speech
            # 750ms / 30ms/frame = 25 frames
            SILENCE_FRAMES_TRIGGER = 25

            while True:
                try:
                    frame = stream.read(CHUNK, exception_on_overflow=False)

                    # Check if the frame contains speech
                    is_speech = vad.is_speech(frame, RATE)

                    if is_speech:
                        if not speaking:
                            print("Speech detected...")
                            speaking = True
                        frames.append(frame)
                        silent_frames_count = 0
                    elif not is_speech and speaking:
                        # User was speaking, but is now silent
                        silent_frames_count += 1
                        frames.append(frame)  # Buffer a bit of silence
                        if silent_frames_count > SILENCE_FRAMES_TRIGGER:
                            speaking = False
                            break  # End of speech detected!
                    elif not speaking and len(frames) > 0:
                        # Clear buffer if it's just initial silence
                        frames.clear()

                except IOError as e:
                    print(f"Stream read error: {e}")
                    break  # Exit VAD loop on error

            # We've broken out of the VAD loop, speech has ended.
            stream.stop_stream()
            stream.close()

            # --- Process the buffered audio ---
            if frames:
                audio_data = b"".join(frames)
                frames.clear()

                try:
                    # --- This is the NEW STT part ---
                    # We send the whole audio clip at once
                    audio_input = speech.RecognitionAudio(content=audio_data)
                    config = speech.RecognitionConfig(
                        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                        sample_rate_hertz=RATE,
                        language_code=STT_LANGUAGE,
                    )

                    response = speech_client.recognize(config=config, audio=audio_input)

                    if response.results:
                        transcript = response.results[0].alternatives[0].transcript

                        # --- Now call Groq and TTS (same as before) ---
                        groq_response = call_groq(transcript)
                        audio_out = call_tts(groq_response)
                        play_audio(audio_out)
                    else:
                        print("Could not understand audio.")

                except Exception as e:
                    print(f"Error during STT/LLM/TTS cycle: {e}")

    except KeyboardInterrupt:
        print("\nStopping conversation.")
    finally:
        # Clean up
        p.terminate()
        pygame.mixer.quit()


# --- Run the main loop ---
if __name__ == "__main__":
    main_loop()
