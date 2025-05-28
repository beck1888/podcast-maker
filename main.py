# Imports
import sys
import json
import subprocess
import re
import hashlib
from pathlib import Path
from openai import OpenAI
import os
from pydub import AudioSegment
from datetime import datetime

# Settings
DEBUG = True
VALID_SFX = {
    "laugh_track",
    "incorrect_buzzer",
    "correct_buzzer",
    "audience_clap",
    "audience_boo",
    "audience_cheer"
}

# Uniform debug logging function
def log(message: str) -> None:
    if DEBUG:
        timestamp = datetime.now().strftime("%I:%M:%S %p")
        print(f"{timestamp} [DEBUG] {message}")

# API Key getter from 1Password
def get_openai_api_key_from_op(item_name: str = "OpenAI API Key", field: str = "credential") -> str:
    log("get_openai_api_key_from_op: start")
    try:
        log(f"Running op CLI for item '{item_name}'")
        result = subprocess.run(
            ["op", "item", "get", item_name, "--format", "json"],
            check=True,
            capture_output=True,
            text=True
        )
        item_json = json.loads(result.stdout)
        log("Received JSON from op CLI")

        for field_entry in item_json.get("fields", []):
            if field_entry.get("label") == field:
                log(f"Found field '{field}', returning API key")
                return field_entry.get("value")

        raise ValueError(f"Field '{field}' not found in item '{item_name}'.")

    except subprocess.CalledProcessError as e:
        log(f"Error retrieving item '{item_name}': {e.stderr.strip()}")
        raise RuntimeError(f"Failed to retrieve item '{item_name}': {e.stderr.strip()}")

# Initialize OpenAI client
log('Initializing OpenAI client')
client = OpenAI(api_key=get_openai_api_key_from_op())
log('OpenAI client initialized')

# Method to generate the script
def make_script(topic: str, approx_word_count: int) -> str:
    log(f"make_script: start (topic='{topic}', words={approx_word_count})")
    # Load the prompts
    log("Loading system prompt")
    with open('prompts/system_prompt.txt', 'r') as f:
        system_prompt = f.read()
    log('System prompt loaded')

    log("Loading user prompt")
    with open('prompts/user_prompt.txt', 'r') as f:
        user_prompt = f.read()
    user_prompt = user_prompt.replace('{{TOPIC}}', topic)
    user_prompt = user_prompt.replace('{{WORD_LENGTH}}', str(approx_word_count))
    log('User prompt loaded and placeholders replaced')

    # Use the OpenAI chat completions api to generate the script
    log('Requesting chat completion')
    response = client.chat.completions.create(
        model='gpt-4.1-mini',
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ]
    )
    script = response.choices[0].message.content
    log('Chat completion received')

    result = script or "Error generating podcast!"
    log(f"make_script: end, result length={len(result)}")
    return result

# Helper method for parse_line
def parse_line(line: str) -> dict[str, str]:
    log(f"parse_line: processing line '{line.strip()}'")
    line = line.strip()

    speech_match = re.match(
        r'<\s*Speech\s+name\s*=\s*"(.*?)"\s*>\s*(.*?)\s*<\/\s*Speech\s*>',
        line,
        re.DOTALL | re.IGNORECASE
    )
    sfx_match = re.match(
        r'<\s*SFX\s+name\s*=\s*"(.*?)"\s*\/\s*>',
        line,
        re.IGNORECASE
    )

    if speech_match:
        result = {
            "instruction": "tts",
            "voice": speech_match.group(1).strip(),
            "content": speech_match.group(2).strip(),
            "hash": hashlib.sha256(line.encode()).hexdigest()
        }
        log(f"parse_line: matched TTS instruction -> {result}")
        return result
    elif sfx_match:
        result = {
            "instruction": "sfx",
            "content": sfx_match.group(1).strip()
        }
        log(f"parse_line: matched SFX instruction -> {result}")
        return result
    else:
        log("parse_line: no valid instruction found, skipping")
        return None

# Method to parse the script into instructions
def parse_into_instructions(script: str) -> list[dict[str, str]]:
    log("parse_into_instructions: start")
    lines = script.split('\n')
    log(f"Script split into {len(lines)} lines")

    queue: list[dict[str, str]] = []
    for idx, line in enumerate(lines):
        log(f"parse_into_instructions: parsing line {idx}")
        instruction = parse_line(line)
        if instruction:
            queue.append(instruction)
            log(f"Instruction appended: {instruction}")
    log(f"parse_into_instructions: end, total instructions={len(queue)}")

    # Add our custom items to the queue
    queue.insert(0, 'public/audio/intro.mp3')
    log('Added the intro sound effect to the start of the clip queue')
    queue.append('public/audio/outro.mp3')
    log('Added the outro sound effect to the end of the clip queue')

    return queue

# Method to use OpenAI tts endpoint
def generate_speech(speaker_name: str, speech_content: str, speech_hash: str) -> str:
    log(f"generate_speech: start (speaker='{speaker_name}')")
    voice_map = {"alice": "alloy", "bob": "ash"}
    use_api_voice = voice_map.get(speaker_name, 'alloy')
    log(f"Using API voice '{use_api_voice}' for speaker '{speaker_name}'")

    speech_hash_fp = speech_hash + ".mp3"
    speech_file_path = Path(__file__).parent / "tmp" / speech_hash_fp
    log(f"Output file path set to {speech_file_path}")

    audio_response = client.audio.speech.create(
        model='tts-1',
        voice=use_api_voice,
        input=speech_content
    )
    log("TTS API response received, writing to file")
    audio_response.write_to_file(speech_file_path)
    log(f"Audio written to {speech_file_path}")

    return str(speech_file_path)

# Method to generate the whole podcast audio order
def generate_whole_podcast_order(instructions: list[dict[str, str]]) -> list[str]:
    log("generate_whole_podcast_order: start")
    audio_clip_queue: list[str] = []
    log(f"Total instructions to process: {len(instructions)}")

    for idx, instruction in enumerate(instructions):
        log(f"Processing instruction {idx}: {instruction}")
        task = instruction.get('instruction')
        if task == 'tts':
            file_path = generate_speech(
                speaker_name=instruction['voice'],
                speech_content=instruction['content'],
                speech_hash=instruction['hash']
            )
            audio_clip_queue.append(file_path)
            log(f"Appended TTS clip: {file_path}")
        else:
            log(f"Unsupported instruction '{task}', skipping")
    log("generate_whole_podcast_order: end")
    return audio_clip_queue

# Method to stitch clips together and speed up audio
def stitch_clips(queue: list[str]) -> str:
    log("stitch_clips: start")
    def speed_up(audio: AudioSegment, speed: float) -> AudioSegment:
        log(f"speed_up: original frame_rate={audio.frame_rate}, speed={speed}")
        new_frame_rate = int(audio.frame_rate * speed)
        return audio._spawn(audio.raw_data, overrides={"frame_rate": new_frame_rate}).set_frame_rate(audio.frame_rate)

    out_dir = "out"
    os.makedirs(out_dir, exist_ok=True)
    log(f"Output directory ensured: {out_dir}")

    existing = [f for f in os.listdir(out_dir) if f.endswith(".mp3") and f[:3].isdigit()]
    ids = [int(f[:3]) for f in existing]
    next_id = max(ids) + 1 if ids else 1
    filename = f"{next_id:03}.mp3"
    output_fp = os.path.join(out_dir, filename)
    log(f"Next output file determined: {output_fp}")

    combined = AudioSegment.silent(duration=0)
    pause = AudioSegment.silent(duration=200)

    for i, file_path in enumerate(queue):
        log(f"stitch_clips: processing file {i}: {file_path}")
        clip = AudioSegment.from_mp3(file_path)
        sped_up = speed_up(clip, 1.05)
        combined += sped_up
        if i < len(queue) - 1:
            combined += pause
    
    combined.export(output_fp, format="mp3")
    log(f"Podcast exported to {output_fp}")
    log("stitch_clips: end")
    return output_fp

# Main execution
if __name__ == '__main__':
    log("Main: start")
    topic = input("Enter a topic: ")
    target_length = input("Enter target word count (50-500): ")

    try:
        word_count = int(target_length)
        log('Converted input to int')
    except ValueError:
        log('Failed to convert the user\'s target length to an integer')
        sys.exit(1)

    script = make_script(topic, word_count)
    instructions = parse_into_instructions(script)
    clips = generate_whole_podcast_order(instructions)
    podcast_path = stitch_clips(clips)
    print(podcast_path)
    log(f"Main: finished, podcast available at {podcast_path}")
