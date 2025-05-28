# Imports
import json
import subprocess
import re
import hashlib
from pathlib import Path
from openai import OpenAI
import os
from pydub import AudioSegment


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


# Easy debug logging function
def log(message: str) -> None:
    if DEBUG is True:
        print(message)

# API Key getter from 1Password
def get_openai_api_key_from_op(item_name: str="OpenAI API Key", field: str="credential"):
    log('Fetching OpenAI API Key via 1Password')
    try:
        # Run the op CLI command to get the item as JSON
        result = subprocess.run(
            ["op", "item", "get", item_name, "--format", "json"],
            check=True,
            capture_output=True,
            text=True
        )
        item_json = json.loads(result.stdout)

        # Search for the field by label
        for field_entry in item_json.get("fields", []):
            if field_entry.get("label") == field:
                return field_entry.get("value")

        raise ValueError(f"Field '{field}' not found in item '{item_name}'.")

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to retrieve item '{item_name}': {e.stderr.strip()}")

# Global podcast client
log('Creating OpenAI client')
client = OpenAI(api_key=get_openai_api_key_from_op())
log('Created OpenAI client')

# Method to generate the script
def make_script(topic: str, approx_word_count: int) -> str:
    # Load the prompts
    with open('prompts/system_prompt.txt', 'r') as f:
        system_prompt = f.read()
        log('Loaded system prompt')

    with open('prompts/user_prompt.txt', 'r') as f:
        user_prompt = f.read()
        # Replace the topic and length placeholders
        user_prompt = user_prompt.replace('{{TOPIC}}', topic)
        user_prompt = user_prompt.replace('{{WORD_LENGTH}}', str(approx_word_count))
        log('Loaded user prompt and replaced placeholders')

    # Use the OpenAI chat completions api to generate the script
    log('Creating script with api')
    script = client.chat.completions.create(
        model='gpt-4.1-mini',
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    ).choices[0].message.content
    log('Script created')

    return script or "Error generating podcast!"

# Helper method for parse line
def parse_line(line: str) -> dict[str, str]:
    line = line.strip()

    # Improved Speech tag regex: tolerant of extra spaces, tabs, etc.
    speech_match = re.match(
        r'<\s*Speech\s+name\s*=\s*"(.*?)"\s*>\s*(.*?)\s*</\s*Speech\s*>',
        line,
        re.DOTALL | re.IGNORECASE
    )

    # Improved SFX tag regex: tolerant of extra spaces, tabs, etc.
    sfx_match = re.match(
        r'<\s*SFX\s+name\s*=\s*"(.*?)"\s*/\s*>',
        line,
        re.IGNORECASE
    )

    if speech_match:
        return {
            "instruction": "tts",
            "voice": speech_match.group(1).strip(),
            "content": speech_match.group(2).strip(),
            "hash": hashlib.sha256(line.encode()).hexdigest() # For consistent audio file naming
        }
    elif sfx_match:
        return {
            "instruction": "sfx",
            "content": sfx_match.group(1).strip()
        }
    else:
        pass # Skip invalid instructions

# Method to parse the script into the instruction
def parse_into_instructions(script: str) -> list[dict[str, str]]:
    # Split the instructions
    lines = script.split('\n') # New instruction on each line per system prompt

    # Create a queue for the instructions
    queue: list[dict[str, str]] = []

    # Add each tag to the queue
    for line in lines:
        queue.append(parse_line(line))

    return queue

# Method to use OpenAI tts endpoint
def generate_speech(speaker_name: str, speech_content: str, speech_instruction_hash_for_fp: str) -> str:
    # Map the script voice names to actual voices in the endpoint
    voice_map = {
        # Defaults
        "alice": "alloy",
        "bob": "ash"
    }
    use_api_voice = voice_map[speaker_name]

    # Get the file_path to save the voice to
    speech_instruction_hash_for_fp = speech_instruction_hash_for_fp + ".mp3" # Ensure correct extension
    speech_file_path = Path(__file__).parent / "tmp" / speech_instruction_hash_for_fp

    # Create the audio clip
    audio_response = client.audio.speech.create(
        model='tts-1',
        voice=use_api_voice,
        input=speech_content
    )

    # Write the audio clip to the file
    audio_response.write_to_file(speech_file_path)

    # Return the fp the audio was written to
    return speech_file_path.__str__()


# Method to gen the whole podcast audio order and parts
def generate_whole_podcast_order(instructions: list[dict[str, str]]) -> list[str]:
    # Track the audio clips in order they will be played
    audio_clip_queue: list[str] = []


    for instruction in instructions:
        # Get the task type
        task = instruction['instruction']

        if task == 'tts':
            # Gen speech
            audio_clip_queue.append(
                generate_speech(
                    speaker_name=instruction['voice'],
                    speech_content=instruction['content'],
                    speech_instruction_hash_for_fp=instruction['hash']
                )
            )
        else:
            log(f'Error: {task} is not a valid instruction... skipping...')

    return audio_clip_queue

def stitch_clips(queue: list[str]) -> str:
    def speed_up(audio: AudioSegment, speed: float) -> AudioSegment:
        new_frame_rate = int(audio.frame_rate * speed)
        return audio._spawn(audio.raw_data, overrides={"frame_rate": new_frame_rate}).set_frame_rate(audio.frame_rate)

    # Ensure output directory exists
    out_dir = "out"
    os.makedirs(out_dir, exist_ok=True)

    # Find the next ID based on existing files in out/
    existing = [f for f in os.listdir(out_dir) if f.endswith(".mp3") and f[:3].isdigit()]
    ids = [int(f[:3]) for f in existing]
    next_id = max(ids) + 1 if ids else 1
    filename = f"{next_id:03}.mp3"
    output_fp = os.path.join(out_dir, filename)

    # Start stitching
    combined = AudioSegment.silent(duration=0)
    pause = AudioSegment.silent(duration=200)  # 0.2 second pause

    for i, file_path in enumerate(queue):
        clip = AudioSegment.from_mp3(file_path)
        sped_up = speed_up(clip, 1.05) # Speed up the actual voice by 1.05 so it's less slow
        combined += sped_up
        if i < len(queue) - 1:
            combined += pause

    # Export to the output file
    combined.export(output_fp, format="mp3")
    return output_fp



if __name__ == '__main__':
    script = make_script('How to make Pizza', 50)
    instructions = parse_into_instructions(script)
    clips = generate_whole_podcast_order(instructions)
    podcast = stitch_clips(clips)
    print(podcast)
