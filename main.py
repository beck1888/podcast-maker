# Imports
import json
import subprocess
import re
import hashlib
from typing import Any
from pathlib import Path
from terminal import spinner, Item
from openai import OpenAI
import os
from pydub import AudioSegment
from datetime import datetime
from send2trash import send2trash

# Settings
KEEP_LOGS: bool = True
AI_MODEL_FOR_WRITING_SCRIPT: str = 'gpt-4.1'
SPEECH_SPEED_MULTIPLIER: float = 1.04
DELAY_BETWEEN_CLIPS_IN_MS: float = 200
LANGUAGE: str = 'English'

def log_api_call_cost(model: str, token_type: str, content_length: int):
    with open('tmp/api_calls.json', 'r') as f:
        api_call_data: list[dict[str, Any]] = json.load(f)

    api_call_data.append(
        {
            'model': model,
            'type': token_type,
            'length': content_length
        }
    )

    with open('tmp/api_calls.json', 'w') as f:
        json.dump(api_call_data, f, indent=4)

    log('updated api call cost log')

def calculate_api_call_cost_sum_for_run():
    with open('tmp/api_calls.json', 'r') as f:
        api_call_data: list[dict[str, Any]] = json.load(f)

    log(f'opened api call data to sum up, found {str(len(api_call_data))} entries')

    sum = 0

    for call in api_call_data:
        model = call['model']
        token_type = call['type']
        length = call['length']

        log(f"Processing API call: model={model}, type={token_type}, length={length}")

        if model == 'tts-1':
            cost_per_token = 0.000015
            sum += length * cost_per_token
            log(f"Added cost for model 'tts-1': {length * cost_per_token}")

        elif model == 'gpt-4.1-nano':
            tokens = length / 3 # approx 3 chars per token because of dense-sih xml
            log(f"Calculated tokens for 'gpt-4.1-nano': {tokens}")
            if token_type == 'input':
                cost_per_token = 0.0000001
            elif token_type == 'output':
                cost_per_token = 0.0000004
            
            sum += tokens * cost_per_token
            log(f"Added cost for model 'gpt-4.1-nano': {tokens * cost_per_token}")

        elif model == 'gpt-4.1':
            tokens = length / 3 # approx 3 chars per token because of dense-sih xml
            log(f"Calculated tokens for 'gpt-4.1': {tokens}")
            if token_type == 'input':
                cost_per_token = 0.000002
            elif token_type == 'output':
                cost_per_token = 0.000008

            sum += tokens * cost_per_token
            log(f"Added cost for model 'gpt-4.1': {tokens * cost_per_token}")

        else:
            log(f"Invalid API call found, not added to sum: {str(call)}")


    sum = round(sum, 2)
    log(f'estimated run cost is {str(sum)} dollars')
    return sum

# Uniform debug logging function
def log(message: str) -> None:
    if KEEP_LOGS:
        with open('logs.txt', 'a') as f:
            if message == "//space_out_logs;":
                f.write('\n'*2)
            else:
                timestamp = datetime.now().strftime("%I:%M:%S %p")
                f.write(f"{timestamp} [DEBUG] {message}\n")
log('//space_out_logs;')
log("STARTING NEW LOG")

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
def make_script(topic: str, lang: str) -> str:
    log(f"make_script: start (topic='{topic}')")
    # Load the prompts
    log("Loading system prompt")
    with open('prompts/system_prompt.txt', 'r') as f:
        system_prompt = f.read()
    log('System prompt loaded')

    log("Loading user prompt")
    with open('prompts/user_prompt.txt', 'r') as f:
        user_prompt = f.read()
    user_prompt = user_prompt.replace('{{TOPIC}}', topic)
    log('User prompt loaded and placeholder replaced')

    if lang.lower() == 'english' or lang.lower() == "" or not lang.lower:
        log('make_script: English or non-specified version requested, skipping adding any instructions about translation.')
    else:
        log(f"make_script: Non-english language detected in call: (lang='{lang}'). Appending translation instruction.")
        user_prompt += f' Although the tags should stay in English, write JUST the spoken content in {lang} while keeping all non-spoken content such as tags in English and as specified in the system instructions.'

    # Assemble api messages
    _messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ]
    
    length_of_input_tokens = len(system_prompt) + len(user_prompt)
    log_api_call_cost(
        model=AI_MODEL_FOR_WRITING_SCRIPT,
        token_type='input',
        content_length=length_of_input_tokens
    )

    # Log the system messages in logs for easier debugging
    log('make_script: logging script wiring api messages next')
    log(str(_messages))
    log('make_script: ended logging script writing api messages')

    # Use the OpenAI chat completions api to generate the script
    log('make_script: Requesting chat completion')
    response = client.chat.completions.create(
        model=AI_MODEL_FOR_WRITING_SCRIPT,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ]
    )
    script = response.choices[0].message.content
    log('make_script: Chat completion received')

    log_api_call_cost(
        model=AI_MODEL_FOR_WRITING_SCRIPT,
        token_type='output',
        content_length=len(script)
    )

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
    lines = script.split(';') # Separate by semicolon instead
    log(f"Script split into {len(lines)} lines")

    queue: list[dict[str, str]] = []
    for idx, line in enumerate(lines):
        log(f"parse_into_instructions: parsing line {idx}")
        instruction = parse_line(line)
        if instruction:
            queue.append(instruction)
            log(f"Instruction appended: {instruction}")
    log(f"parse_into_instructions: end, total instructions={len(queue)}")


    return queue

# Method to use OpenAI tts endpoint
def generate_speech(speaker_name: str, speech_content: str, speech_hash: str) -> str:
    log(f"generate_speech: start (speaker='{speaker_name}')")
    voice_map = {"emma": "nova", "charlie": "echo", "kate": "shimmer"}
    use_api_voice = voice_map.get(speaker_name, 'alloy') # 'alloy' is a fallback
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

    log_api_call_cost(
        model='tts-1',
        token_type='text',
        content_length=len(speech_content)
    )

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

    # Add our custom items to the queue
    audio_clip_queue.insert(0, 'public/audio/STATIC_intro.mp3')
    log('Added the intro sound effect to the start of the clip queue')
    audio_clip_queue.append('public/audio/STATIC_outro.mp3')
    log('Added the outro sound effect to the end of the clip queue')


    return audio_clip_queue

# Method to stitch clips together and speed up audio
def stitch_clips(queue: list[str], podcast_name_to_use: str, speed: float, delay_between_clips: float) -> str:
    log("stitch_clips: start")

    log(f'stich_clips: received params (podcast_name_to_use={podcast_name_to_use}, speed={str(speed)}, delay={str(delay_between_clips)})')

    def speed_up(audio: AudioSegment, speed: float) -> AudioSegment:
        log(f"speed_up: original frame_rate={audio.frame_rate}, speed={speed}")
        new_frame_rate = int(audio.frame_rate * speed)
        return audio._spawn(audio.raw_data, overrides={"frame_rate": new_frame_rate}).set_frame_rate(audio.frame_rate)

    out_dir = "out"
    os.makedirs(out_dir, exist_ok=True)
    log(f"Output directory ensured: {out_dir}")

    filename = f"{podcast_name_to_use}.mp3"
    output_fp = os.path.join(out_dir, filename)
    log(f"Output file determined: {output_fp}")

    combined = AudioSegment.silent(duration=0)
    pause = AudioSegment.silent(duration=delay_between_clips)

    for i, file_path in enumerate(queue):
        log(f"stitch_clips: processing file {i}: {file_path}")
        clip = AudioSegment.from_mp3(file_path)

        basename = os.path.basename(file_path)
        if basename.startswith("STATIC_"):
            log(f"Skipping speed-up for STATIC clip: {basename}")
            processed_clip = clip
        else:
            processed_clip = speed_up(clip, speed)
            log(f"Applied speed-up to: {basename}")

        combined += processed_clip
        if i < len(queue) - 1:
            combined += pause

    combined.export(output_fp, format="mp3")
    log(f"Podcast exported to {output_fp}")
    log("stitch_clips: end")
    return output_fp

# Use AI to rename to file so it's easier to find
def use_ai_to_gen_podcast_filename(topic: str) -> str:
    log('use_ai_to_gen_podcast_filename: start')
    naming_prompt: str = f"A person has just gotten done making a podcast about this topic: {topic}. Please give the file name for the podcast a clear and super short title of 2-3 words so it's easy to find. Use title case. Return just the file name and nothing else (no extensions, etc). Prioritize clarity and specificity over grammar."

    log_api_call_cost('gpt-4.1-nano', 'input', len(naming_prompt))

    log('use_ai_to_gen_podcast_filename: calling api')
    new_title = client.chat.completions.create(
        model='gpt-4.1-nano',
        messages=[
            {
                "role": "user",
                "content": naming_prompt
            }
        ]
    ).choices[0].message.content

    log_api_call_cost('gpt-4.1-nano', 'output', len(str(new_title)))


    log(f'use_ai_to_gen_podcast_filename: AI came up with the filename: {new_title}')

    new_title = str(new_title).removeprefix('.mp3') # Just in-case

    log('use_ai_to_gen_podcast_filename: end')
    return new_title or topic.title() # Return the new title name (sanitized just in case), or, as a fallback, the prompt


# Ensure needed files exist in workspace
def set_up():
    for dirname in ['out', 'tmp']:
        log(f'Ensuring the {dirname} directory exists')
        os.makedirs(dirname, exist_ok=True)
    
    with open('tmp/api_calls.json', 'w') as f:
        json.dump([], f)

# Clean up the tmp directory as we're done with it
def clean_up():
    send2trash('tmp')
    log('removed the tmp directory')
    client.close()
    log('closed the openai client')
    log("ENDING LOG RUN")
    log('//space_out_logs;')

# Main execution
if __name__ == '__main__':
    log("Main: start")

    set_up()

    topic = input("Enter a topic: ")

    with spinner("Writing script"):
        script = make_script(topic, lang=LANGUAGE) # log token cost = true

    with spinner("Generating title"):
        ai_generated_title = use_ai_to_gen_podcast_filename(topic) # # log token cost = true

    with spinner("Parsing script"):
        instructions = parse_into_instructions(script) # log token cost = skip, n/a

    with spinner("Generating audio clips"):
        clips = generate_whole_podcast_order(instructions)

    with spinner("Stitching clips"): # log token cost = skip, n/a
        podcast_path = stitch_clips(
            queue=clips,
            podcast_name_to_use=ai_generated_title,
            speed=SPEECH_SPEED_MULTIPLIER,
            delay_between_clips=DELAY_BETWEEN_CLIPS_IN_MS
        )

    Item.checked(f"Podcast available at: '\033[33m{podcast_path}\033[0m'", indent=False) # Print final path in Yellow

    cost = calculate_api_call_cost_sum_for_run()
    
    log(f"Main: finished, podcast available at {podcast_path} | Cost: {str(cost)} dollars")

    ...

    clean_up()
