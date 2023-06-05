from typing import Union

from argparse import ArgumentParser
from pathlib import Path
import subprocess
import librosa
import os
import time
import random

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import *
from moviepy.video.io.VideoFileClip import VideoFileClip

from moviepy.editor import VideoFileClip, AudioFileClip


import openai
import uuid
import tempfile
import shlex
import shutil
from utils import format_bash_command

allowed_medias = [".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif", ".svg", ".mp3", ".wav", ".ogg", ".mp4",
                  ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm", ".mpg", ".mpeg", ".m4v", ".3gp", ".3g2", ".3gpp"]


import asyncio
import json
import hashlib
from os import path, getenv
from pydub import AudioSegment

import gradio as gr

import torch

import edge_tts

from datetime import datetime
from scipy.io.wavfile import write

import config
import util
from infer_pack.models import (
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono
)
from vc_infer_pipeline import VC
    
# Reference: https://huggingface.co/spaces/zomehwh/rvc-models/blob/main/app.py#L21  # noqa
in_hf_space = getenv('SYSTEM') == 'spaces'

high_quality = True

# Argument parsing
arg_parser = ArgumentParser()
arg_parser.add_argument(
    '--hubert',
    default=getenv('RVC_HUBERT', 'hubert_base.pt'),
    help='path to hubert base model (default: hubert_base.pt)'
)
arg_parser.add_argument(
    '--config',
    default=getenv('RVC_MULTI_CFG', 'multi_config.json'),
    help='path to config file (default: multi_config.json)'
)
arg_parser.add_argument(
    '--api',
    action='store_true',
    help='enable api endpoint'
)
arg_parser.add_argument(
    '--cache-examples',
    action='store_true',
    help='enable example caching, please remember delete gradio_cached_examples folder when example config has been modified'  # noqa
)
args = arg_parser.parse_args()

app_css = '''
#model_info img {
    max-width: 100px;
    max-height: 100px;
    float: right;
}

#model_info p {
    margin: unset;
}
'''

app = gr.Blocks(
    theme=gr.themes.Soft(primary_hue="orange", secondary_hue="slate"),
    css=app_css,
    analytics_enabled=False
)

# Load hubert model
hubert_model = util.load_hubert_model(config.device, args.hubert)
hubert_model.eval()

# Load models
multi_cfg = json.load(open(args.config, 'r'))
loaded_models = []

for model_name in multi_cfg.get('models'):
    print(f'Loading model: {model_name}')

    # Load model info
    model_info = json.load(
        open(path.join('model', model_name, 'config.json'), 'r')
    )

    # Load RVC checkpoint
    cpt = torch.load(
        path.join('model', model_name, model_info['model']),
        map_location='cpu'
    )
    tgt_sr = cpt['config'][-1]
    cpt['config'][-3] = cpt['weight']['emb_g.weight'].shape[0]  # n_spk

    if_f0 = cpt.get('f0', 1)
    net_g: Union[SynthesizerTrnMs768NSFsid, SynthesizerTrnMs768NSFsid_nono]
    if if_f0 == 1:
        net_g = SynthesizerTrnMs768NSFsid(
            *cpt['config'],
            is_half=util.is_half(config.device)
        )
    else:
        net_g = SynthesizerTrnMs768NSFsid_nono(*cpt['config'])

    del net_g.enc_q

    # According to original code, this thing seems necessary.
    print(net_g.load_state_dict(cpt['weight'], strict=False))

    net_g.eval().to(config.device)
    net_g = net_g.half() if util.is_half(config.device) else net_g.float()

    vc = VC(tgt_sr, config)
    
    loaded_models.append(dict(
        name=model_name,
        metadata=model_info,
        vc=vc,
        net_g=net_g,
        if_f0=if_f0,
        target_sr=tgt_sr
    ))
        
print(f'Models loaded: {len(loaded_models)}')

# Edge TTS speakers
tts_speakers_list = asyncio.get_event_loop().run_until_complete(edge_tts.list_voices())  # noqa


# API key

def access(apikey):
    os.environ["OPENAI_API_KEY"] = apikey
    openai.api_key = os.environ["OPENAI_API_KEY"]
    return "填写成功"

# ChatGPT powered

def get_files_infos(files):
    results = []
    for file in files:
        file_path = Path(file.name)
        info = {}
        info["size"] = os.path.getsize(file_path)
        info["name"] = file_path.name
        file_extension = file_path.suffix

        if file_extension in (".mp4", ".avi", ".mkv", ".mov"):
            info["type"] = "video"
            video = VideoFileClip(file.name)
            info["duration"] = video.duration
            info["dimensions"] = "{}x{}".format(video.size[0], video.size[1])
            if video.audio:
                info["type"] = "video/audio"
                info["audio_channels"] = video.audio.nchannels
            video.close()
        elif file_extension in (".mp3", ".wav"):
            info["type"] = "audio"
            audio = AudioFileClip(file.name)
            info["duration"] = audio.duration
            info["audio_channels"] = audio.nchannels
            audio.close()
        elif file_extension in (
            ".png",
            ".jpg",
            ".jpeg",
            ".tiff",
            ".bmp",
            ".gif",
            ".svg",
        ):
            info["type"] = "image"
            img = Image.open(file.name)
            info["dimensions"] = "{}x{}".format(img.size[0], img.size[1])
        results.append(info)
    return results


def get_completion(prompt, files_info, top_p, temperature):

    files_info_string = ""
    for file_info in files_info:
        files_info_string += f"""{file_info["type"]} {file_info["name"]}"""
        if file_info["type"] == "video" or file_info["type"] == "image":
            files_info_string += f""" {file_info["dimensions"]}"""
        if file_info["type"] == "video" or file_info["type"] == "audio":
            files_info_string += f""" {file_info["duration"]}s"""
        if file_info["type"] == "audio" or file_info["type"] == "video/audio":
            files_info_string += f""" {file_info["audio_channels"]} audio channels"""
        files_info_string += "\n"

    messages = [
        {
            "role": "system",
            # "content": f"""Act as a FFMPEG expert. Create a valid FFMPEG command that will be directly pasted in the terminal. Using those files: {files_info} create the FFMPEG command to achieve this: "{prompt}". Make sure it's a valid command that will not do any error. Always name the output of the FFMPEG command "output.mp4". Always use the FFMPEG overwrite option (-y). Don't produce video longer than 1 minute. Think step by step but never give any explanation, only the shell command.""",
            # "content": f"""You'll need to create a valid FFMPEG command that will be directly pasted in the terminal. You have those files (images, videos, and audio) at your disposal: {files_info} and you need to compose a new video using FFMPEG and following those instructions: "{prompt}". You'll need to use as many assets as you can. Make sure it's a valid command that will not do any error. Always name the output of the FFMPEG command "output.mp4". Always use the FFMPEG overwrite option (-y). Try to avoid using -filter_complex option.  Don't produce video longer than 1 minute. Think step by step but never give any explanation, only the shell command.""",
            "content": f"""
You are a very experienced agent controlling a UNIX terminal and a contributor to the ffmpeg project. You are given:
(1) a set of video, audio and/or image assets. Including their name, duration, dimensions and file size
(2) the description of a new video you need to create from the list of assets
Based on the available assets and the description, your objective issue a FFMPEG to create a new video using the assets.
This will often involve putting assets one after the other, cropping the video format, or playing music in the background. Avoid using complex FFMPEG options, and try to keep the command as simple as possible as it will be directly paster into the terminal.
Always output the media a video/mp4 and output file "output.mp4". Provide only the shell command without any explanations.
The current assets and objective follow. Reply with the FFMPEG command:
AVAILABLE ASSETS LIST:
{files_info_string}
OBJECTIVE: {prompt}
YOUR FFMPEG COMMAND:""",
        }
    ]

    print(messages[0]["content"])

    try:
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                                  messages=messages,
                                                  top_p=top_p,
                                                  temperature=temperature)

        command = completion.choices[0].message.content.replace("\n", "")

        # remove output.mp4 with the actual output file path
        command = command.replace("output.mp4", "")

        return command
    except Exception as e:
        print("FROM OPENAI", e)
        raise Exception("OpenAI API error")


def update(files, prompt, top_p=1, temperature=1):
    if prompt == "":
        raise gr.Error("Please enter a prompt.")

    files_info = get_files_infos(files)
    # disable this if you're running the app locally or on your own server
    for file_info in files_info:
        if file_info["type"] == "video":
            if file_info["duration"] > 1000:
                raise gr.Error(
                    "Please make sure all videos are less than 2 minute long."
                )
        if file_info["size"] > 100000000:
            raise gr.Error(
                "Please make sure all files are less than 10MB in size."
            )
    try:
        command_string = get_completion(prompt, files_info, top_p, temperature)
        print(
            f"""\n\n/// START OF COMMAND ///:\n\n{command_string}\n\n/// END OF COMMAND ///\n\n""")

        # split command string into list of arguments
        args = shlex.split(command_string)
        if (args[0] != "ffmpeg"):
            raise Exception("Command does not start with ffmpeg")
        temp_dir = tempfile.mkdtemp()
        # copy files to temp dir
        for file in files:
            file_path = Path(file.name)
            shutil.copy(file_path, temp_dir)

        # test if ffmpeg command is valid dry run
        ffmpg_dry_run = subprocess.run(
            args + ["-f", "null", "-"], stderr=subprocess.PIPE, text=True, cwd=temp_dir)
        if ffmpg_dry_run.returncode == 0:
            print("Command is valid.")
        else:
            print("Command is not valid. Error output:")
            print(ffmpg_dry_run.stderr)
            raise Exception(
                "FFMPEG generated command is not valid. Please try again.")

        output_file_name = f'output_{uuid.uuid4()}.mp4'
        output_file_path = str((Path(temp_dir) / output_file_name).resolve())
        subprocess.run(args + ["-y", output_file_path], cwd=temp_dir)
        generated_command = f"### Generated Command\n```bash\n{format_bash_command(args)}\n    -y output.mp4\n```"
        return output_file_path, gr.update(value=generated_command)
    except Exception as e:
        print("FROM UPDATE", e)
        raise gr.Error(e)



# Make MV
def make_bars_image(height_values, index, new_height):
    
    # Define the size of the image
    width = 512  
    height = new_height
    
    # Create a new image with a transparent background
    image = Image.new('RGBA', (width, height), color=(0, 0, 0, 0))
    
    # Get the image drawing context
    draw = ImageDraw.Draw(image)
    
    # Define the rectangle width and spacing
    rect_width = 2
    spacing = 2
    
    # Define the list of height values for the rectangles
    #height_values = [20, 40, 60, 80, 100, 80, 60, 40]
    num_bars = len(height_values)
    # Calculate the total width of the rectangles and the spacing
    total_width = num_bars * rect_width + (num_bars - 1) * spacing
    
    # Calculate the starting position for the first rectangle
    start_x = int((width - total_width) / 2)
    # Define the buffer size
    buffer_size = 80
    # Draw the rectangles from left to right
    x = start_x
    for i, height in enumerate(height_values):
        
        # Define the rectangle coordinates
        y0 = buffer_size
        y1 = height + buffer_size
        x0 = x
        x1 = x + rect_width

        # Draw the rectangle
        draw.rectangle([x0, y0, x1, y1], fill='white')  
        
        # Move to the next rectangle position
        if i < num_bars - 1:
            x += rect_width + spacing
        

    # Rotate the image by 180 degrees
    image = image.rotate(180)
    
    # Mirror the image
    image = image.transpose(Image.FLIP_LEFT_RIGHT)
    
    # Save the image
    image.save('audio_bars_'+ str(index) + '.png')

    return 'audio_bars_'+ str(index) + '.png'

def db_to_height(db_value):
    # Scale the dB value to a range between 0 and 1
    scaled_value = (db_value + 80) / 80
    
    # Convert the scaled value to a height between 0 and 100
    height = scaled_value * 50
    
    return height

def infer(title, audio_in, image_in):
    # Load the audio file
    audio_path = audio_in
    audio_data, sr = librosa.load(audio_path)

    # Get the duration in seconds
    duration = librosa.get_duration(y=audio_data, sr=sr)
    
    # Extract the audio data for the desired time
    start_time = 0 # start time in seconds
    end_time = duration # end time in seconds
    
    start_index = int(start_time * sr)
    end_index = int(end_time * sr)
    
    audio_data = audio_data[start_index:end_index]
    
    # Compute the short-time Fourier transform
    hop_length = 512

    
    stft = librosa.stft(audio_data, hop_length=hop_length)
    spectrogram = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

    # Get the frequency values
    freqs = librosa.fft_frequencies(sr=sr, n_fft=stft.shape[0])

    # Select the indices of the frequency values that correspond to the desired frequencies
    n_freqs = 114
    freq_indices = np.linspace(0, len(freqs) - 1, n_freqs, dtype=int)
    
    # Extract the dB values for the desired frequencies
    db_values = []
    for i in range(spectrogram.shape[1]):
        db_values.append(list(zip(freqs[freq_indices], spectrogram[freq_indices, i])))
    
    # Print the dB values for the first time frame
    print(db_values[0])

    proportional_values = []

    for frame in db_values:
        proportional_frame = [db_to_height(db) for f, db in frame]
        proportional_values.append(proportional_frame)

    print(proportional_values[0])
    print("AUDIO CHUNK: " + str(len(proportional_values)))

    # Open the background image
    background_image = Image.open(image_in)
    
    # Resize the image while keeping its aspect ratio
    bg_width, bg_height = background_image.size
    aspect_ratio = bg_width / bg_height
    new_width = 512
    new_height = int(new_width / aspect_ratio)
    resized_bg = background_image.resize((new_width, new_height))

    # Apply black cache for better visibility of the white text
    bg_cache = Image.open('black_cache.png')
    resized_bg.paste(bg_cache, (0, resized_bg.height - bg_cache.height), mask=bg_cache)

    # Create a new ImageDraw object
    draw = ImageDraw.Draw(resized_bg)
    
    # Define the text to be added
    text = title
    font = ImageFont.truetype("Lato-Regular.ttf", 16)
    text_color = (255, 255, 255) # white color
    
    # Calculate the position of the text
    text_width, text_height = draw.textsize(text, font=font)
    x = 30
    y = new_height - 70
    
    # Draw the text on the image
    draw.text((x, y), text, fill=text_color, font=font)

    # Save the resized image
    resized_bg.save('resized_background.jpg')
    
    generated_frames = []
    for i, frame in enumerate(proportional_values): 
        bars_img = make_bars_image(frame, i, new_height)
        bars_img = Image.open(bars_img)
        # Paste the audio bars image on top of the background image
        fresh_bg = Image.open('resized_background.jpg')
        fresh_bg.paste(bars_img, (0, 0), mask=bars_img)
        # Save the image
        fresh_bg.save('audio_bars_with_bg' + str(i) + '.jpg')
        generated_frames.append('audio_bars_with_bg' + str(i) + '.jpg')
    print(generated_frames)

    # Create a video clip from the images
    clip = ImageSequenceClip(generated_frames, fps=len(generated_frames)/(end_time-start_time))
    audio_clip = AudioFileClip(audio_in)
    clip = clip.set_audio(audio_clip)
    # Set the output codec
    codec = 'libx264'
    audio_codec = 'aac'
    # Save the video to a file
    clip.write_videofile("my_video.mp4", codec=codec, audio_codec=audio_codec)

    retimed_clip = VideoFileClip("my_video.mp4")

    # Set the desired frame rate
    new_fps = 25
    
    # Create a new clip with the new frame rate
    new_clip = retimed_clip.set_fps(new_fps)
    
    # Save the new clip as a new video file
    new_clip.write_videofile("my_video_retimed.mp4", codec=codec, audio_codec=audio_codec)

    return "my_video_retimed.mp4"

# mix vocal and non-vocal
def mix(audio1, audio2):
  sound1 = AudioSegment.from_file(audio1)
  sound2 = AudioSegment.from_file(audio2)
  length = len(sound1)
  mixed = sound1[:length].overlay(sound2)

  mixed.export("song.wav", format="wav")

  return "song.wav"

# Bilibili
def youtube_downloader(
    video_identifier,
    start_time,
    end_time,
    output_filename="track.wav",
    num_attempts=5,
    url_base="",
    quiet=False,
    force=True,
):
    output_path = Path(output_filename)
    if output_path.exists():
        if not force:
            return output_path
        else:
            output_path.unlink()

    quiet = "--quiet --no-warnings" if quiet else ""
    command = f"""
        yt-dlp {quiet} -x --audio-format wav -f bestaudio -o "{output_filename}" --download-sections "*{start_time}-{end_time}" "{url_base}{video_identifier}"  # noqa: E501
    """.strip()

    attempts = 0
    while True:
        try:
            _ = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError:
            attempts += 1
            if attempts == num_attempts:
                return None
        else:
            break

    if output_path.exists():
        return output_path
    else:
        return None

def audio_separated(audio_input, progress=gr.Progress()):
    # start progress
    progress(progress=0, desc="Starting...")
    time.sleep(0.1)

    # check file input
    if audio_input is None:
        # show progress
        for i in progress.tqdm(range(100), desc="Please wait..."):
            time.sleep(0.01)
            
        return (None, None, 'Please input audio.')

    # create filename
    filename = str(random.randint(10000,99999))+datetime.now().strftime("%d%m%Y%H%M%S")
    
    # progress
    progress(progress=0.10, desc="Please wait...")
    
    # make dir output
    os.makedirs("output", exist_ok=True)
    
    # progress
    progress(progress=0.20, desc="Please wait...")
    
    # write
    if high_quality:
        write(filename+".wav", audio_input[0], audio_input[1])
    else:
        write(filename+".mp3", audio_input[0], audio_input[1])
        
    # progress
    progress(progress=0.50, desc="Please wait...")

    # demucs process
    if high_quality:
        command_demucs = "python3 -m demucs --two-stems=vocals -d cpu "+filename+".wav -o output"
    else:
        command_demucs = "python3 -m demucs --two-stems=vocals --mp3 --mp3-bitrate 128 -d cpu "+filename+".mp3 -o output"
    
    os.system(command_demucs)
    
    # progress
    progress(progress=0.70, desc="Please wait...")
    
    # remove file audio
    if high_quality:
        command_delete = "rm -v ./"+filename+".wav"
    else:
        command_delete = "rm -v ./"+filename+".mp3"
    
    os.system(command_delete)
    
    # progress
    progress(progress=0.80, desc="Please wait...")
    
    # progress
    for i in progress.tqdm(range(80,100), desc="Please wait..."):
        time.sleep(0.1)

    if high_quality:
        return "./output/htdemucs/"+filename+"/vocals.wav","./output/htdemucs/"+filename+"/no_vocals.wav","Successfully..."
    else:
        return "./output/htdemucs/"+filename+"/vocals.mp3","./output/htdemucs/"+filename+"/no_vocals.mp3","Successfully..."

        
# https://github.com/fumiama/Retrieval-based-Voice-Conversion-WebUI/blob/main/infer-web.py#L118  # noqa
def vc_func(
    input_audio, model_index, pitch_adjust, f0_method, feat_ratio,
    filter_radius, rms_mix_rate, resample_option
):
    if input_audio is None:
        return (None, 'Please provide input audio.')

    if model_index is None:
        return (None, 'Please select a model.')

    model = loaded_models[model_index]

    # Reference: so-vits
    (audio_samp, audio_npy) = input_audio

    # https://huggingface.co/spaces/zomehwh/rvc-models/blob/main/app.py#L49
    # Can be change well, we will see
    if (audio_npy.shape[0] / audio_samp) > 600 and in_hf_space:
        return (None, 'Input audio is longer than 600 secs.')

    # Bloody hell: https://stackoverflow.com/questions/26921836/
    if audio_npy.dtype != np.float32:  # :thonk:
        audio_npy = (
            audio_npy / np.iinfo(audio_npy.dtype).max
        ).astype(np.float32)

    if len(audio_npy.shape) > 1:
        audio_npy = librosa.to_mono(audio_npy.transpose(1, 0))

    if audio_samp != 16000:
        audio_npy = librosa.resample(
            audio_npy,
            orig_sr=audio_samp,
            target_sr=16000
        )

    pitch_int = int(pitch_adjust)

    resample = (
        0 if resample_option == 'Disable resampling'
        else int(resample_option)
    )

    times = [0, 0, 0]

    checksum = hashlib.sha512()
    checksum.update(audio_npy.tobytes())

    output_audio = model['vc'].pipeline(
        hubert_model,
        model['net_g'],
        model['metadata'].get('speaker_id', 0),
        audio_npy,
        checksum.hexdigest(),
        times,
        pitch_int,
        f0_method,
        path.join('model', model['name'], model['metadata']['feat_index']),
        feat_ratio,
        model['if_f0'],
        filter_radius,
        model['target_sr'],
        resample,
        rms_mix_rate,
        'v2'
    )

    out_sr = (
        resample if resample >= 16000 and model['target_sr'] != resample
        else model['target_sr']
    )

    print(f'npy: {times[0]}s, f0: {times[1]}s, infer: {times[2]}s')
    return ((out_sr, output_audio), 'Success')


async def edge_tts_vc_func(
    input_text, model_index, tts_speaker, pitch_adjust, f0_method, feat_ratio,
    filter_radius, rms_mix_rate, resample_option
):
    if input_text is None:
        return (None, 'Please provide TTS text.')

    if tts_speaker is None:
        return (None, 'Please select TTS speaker.')

    if model_index is None:
        return (None, 'Please select a model.')

    speaker = tts_speakers_list[tts_speaker]['ShortName']
    (tts_np, tts_sr) = await util.call_edge_tts(speaker, input_text)
    return vc_func(
        (tts_sr, tts_np),
        model_index,
        pitch_adjust,
        f0_method,
        feat_ratio,
        filter_radius,
        rms_mix_rate,
        resample_option
    )


def update_model_info(model_index):
    if model_index is None:
        return str(
            '### Model info\n'
            'Please select a model from dropdown above.'
        )

    model = loaded_models[model_index]
    model_icon = model['metadata'].get('icon', '')

    return str(
        '### Model info\n'
        '![model icon]({icon})'
        '**{name}**\n\n'
        'Author: {author}\n\n'
        'Source: {source}\n\n'
        '{note}'
    ).format(
        name=model['metadata'].get('name'),
        author=model['metadata'].get('author', 'Anonymous'),
        source=model['metadata'].get('source', 'Unknown'),
        note=model['metadata'].get('note', ''),
        icon=(
            model_icon
            if model_icon.startswith(('http://', 'https://'))
            else '/file/model/%s/%s' % (model['name'], model_icon)
        )
    )


def _example_vc(
    input_audio, model_index, pitch_adjust, f0_method, feat_ratio,
    filter_radius, rms_mix_rate, resample_option
):
    (audio, message) = vc_func(
        input_audio, model_index, pitch_adjust, f0_method, feat_ratio,
        filter_radius, rms_mix_rate, resample_option
    )
    return (
        audio,
        message,
        update_model_info(model_index)
    )


async def _example_edge_tts(
    input_text, model_index, tts_speaker, pitch_adjust, f0_method, feat_ratio,
    filter_radius, rms_mix_rate, resample_option
):
    (audio, message) = await edge_tts_vc_func(
        input_text, model_index, tts_speaker, pitch_adjust, f0_method,
        feat_ratio, filter_radius, rms_mix_rate, resample_option
    )
    return (
        audio,
        message,
        update_model_info(model_index)
    )


with app:
    gr.HTML("<center>"
            "<h1>🥳🎶🎡 - AI歌手，RVC歌声转换 + AI变声</h1>"
            "</center>")
    gr.Markdown("### <center>🦄 - 能够自动提取视频中的声音，并去除背景音；Powered by [RVC-Project](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)</center>")
    gr.Markdown("### <center>更多精彩应用，敬请关注[滔滔AI](http://www.talktalkai.com)；滔滔AI，为爱滔滔！💕</center>")

    with gr.Tab("🤗 - B站视频提取声音"):
        with gr.Row():
            with gr.Column():
                ydl_url_input  = gr.Textbox(label="B站视频网址(可直接填写相应的BV号)", value = "https://www.bilibili.com/video/BV...")
                start = gr.Number(value=0, label="起始时间 (秒)")
                end = gr.Number(value=15, label="结束时间 (秒)")
                ydl_url_submit = gr.Button("提取声音文件吧", variant="primary")
                as_audio_submit = gr.Button("去除背景音吧", variant="primary")
            with gr.Column():
                ydl_audio_output = gr.Audio(label="Audio from Bilibili")
                as_audio_input  = ydl_audio_output
                as_audio_vocals    = gr.Audio(label="歌曲人声部分")
                as_audio_no_vocals = gr.Audio(label="Music only", type="filepath", visible=False)
                as_audio_message   = gr.Textbox(label="Message", visible=False)
                
    ydl_url_submit.click(fn=youtube_downloader, inputs=[ydl_url_input, start, end], outputs=[ydl_audio_output])
    as_audio_submit.click(fn=audio_separated, inputs=[as_audio_input], outputs=[as_audio_vocals, as_audio_no_vocals, as_audio_message], show_progress=True, queue=True)

    with gr.Tab("🔮 - ChatGPT音乐视频"):
        gr.Markdown(
            """
                # <span style="margin-right: 0.3rem;">🏞</span>ChatGPT音乐视频
                更多使用案例请参考[Project](https://huggingface.co/spaces/huggingface-projects/video-composer-gpt4)
            """,
            elem_id="header",
        )
        with gr.Row():
            with gr.Column():
                inpchat = gr.Textbox(label="请先填写您的OpenAI API key", type="password")
                user_files = gr.File(
                    file_count="multiple", label="Media files", keep_filename=True,
                    file_types=allowed_medias
                )
                user_prompt = gr.Textbox(
                    value="Make a video with a white waveform of the audio taking all screen space, also add the image as the background",
                    label="Instructions",
                    lines=3
                )
                btnchat = gr.Button("安全提交您的API key吧", variant="primary")
                btnGPT = gr.Button("开始制作专属音乐视频吧", variant="primary")
                with gr.Accordion("Parameters", open=False):
                    top_p = gr.Slider(minimum=-0, maximum=1.0, value=0, step=0.05,
                                      interactive=True, label="Top-p (nucleus sampling)")
                    temperature = gr.Slider(
                        minimum=-0, maximum=5.0, value=0, step=0, interactive=True, label="Temperature")
            with gr.Column():
                outchat = gr.Textbox(label="API key填写状态")
                generated_video = gr.Video(
                    interactive=False, label="Generated Video", include_audio=True
                )
                generated_command = gr.Markdown()

            btnchat.click(fn=access, inputs=[inpchat], outputs=[outchat])
            btnGPT.click(
                fn=update, inputs=[user_files, user_prompt, top_p, temperature],
                outputs=[generated_video, generated_command]
            )
                    
    with gr.Row():
        with gr.Column():
            with gr.Tab('🎶 - 歌声转换'):
                input_audio = as_audio_vocals
                vc_convert_btn = gr.Button('进行歌声转换吧！', variant='primary')
                full_song = gr.Button("加入歌曲伴奏吧！", variant="primary")
                new_song = gr.Audio(label="AI歌手+伴奏", type="filepath")

            with gr.Tab('🎙️ - 文本转语音'):
                tts_input = gr.Textbox(
                    label='请填写您想要转换的文本(中英皆可)',
                    lines=3
                )
                tts_speaker = gr.Dropdown(
                    [
                        '%s (%s)' % (
                            s['FriendlyName'],
                            s['Gender']
                        )
                        for s in tts_speakers_list
                    ],
                    label='请选择一个相应语言的说话人',
                    type='index'
                )

                tts_convert_btn = gr.Button('进行AI变声吧', variant='primary')
                
            with gr.Tab("📺 - 音乐视频"):
                with gr.Row():
                    with gr.Column():
                        inp1 = gr.Textbox(label="为视频配上精彩的文案吧(选填;英文)")
                        inp2 = new_song
                        inp3 = gr.Image(source='upload', type='filepath', label="上传一张背景图片吧")
                        btn = gr.Button("生成您的专属音乐视频吧", variant="primary")
              
                    with gr.Column():
                        out1 = gr.Video(label='您的专属音乐视频')
            btn.click(fn=infer, inputs=[inp1, inp2, inp3], outputs=[out1])
            
            pitch_adjust = gr.Slider(
                label='Pitch',
                minimum=-24,
                maximum=24,
                step=1,
                value=0
            )
            f0_method = gr.Radio(
                label='f0 methods',
                choices=['pm', 'harvest'],
                value='pm',
                interactive=True
            )

            with gr.Accordion('更多设置', open=False):
                feat_ratio = gr.Slider(
                    label='Feature ratio',
                    minimum=0,
                    maximum=1,
                    step=0.1,
                    value=0.6
                )
                filter_radius = gr.Slider(
                    label='Filter radius',
                    minimum=0,
                    maximum=7,
                    step=1,
                    value=3
                )
                rms_mix_rate = gr.Slider(
                    label='Volume envelope mix rate',
                    minimum=0,
                    maximum=1,
                    step=0.1,
                    value=1
                )
                resample_rate = gr.Dropdown(
                    [
                        'Disable resampling',
                        '16000',
                        '22050',
                        '44100',
                        '48000'
                    ],
                    label='Resample rate',
                    value='Disable resampling'
                )

        with gr.Column():
            # Model select
            model_index = gr.Dropdown(
                [
                    '%s - %s' % (
                        m['metadata'].get('source', 'Unknown'),
                        m['metadata'].get('name')
                    )
                    for m in loaded_models
                ],
                label='请选择您的AI歌手(必选)',
                type='index'
            )

            # Model info
            with gr.Box():
                model_info = gr.Markdown(
                    '### AI歌手信息\n'
                    'Please select a model from dropdown above.',
                    elem_id='model_info'
                )

            output_audio = gr.Audio(label='AI歌手(无伴奏)', type="filepath")
            output_msg = gr.Textbox(label='Output message')

    multi_examples = multi_cfg.get('examples')
    if (
        multi_examples and
        multi_examples.get('vc') and multi_examples.get('tts_vc')
    ):
        with gr.Accordion('Sweet sweet examples', open=False):
            with gr.Row():
                # VC Example
                if multi_examples.get('vc'):
                    gr.Examples(
                        label='Audio conversion examples',
                        examples=multi_examples.get('vc'),
                        inputs=[
                            input_audio, model_index, pitch_adjust, f0_method,
                            feat_ratio
                        ],
                        outputs=[output_audio, output_msg, model_info],
                        fn=_example_vc,
                        cache_examples=args.cache_examples,
                        run_on_click=args.cache_examples
                    )

                # Edge TTS Example
                if multi_examples.get('tts_vc'):
                    gr.Examples(
                        label='TTS conversion examples',
                        examples=multi_examples.get('tts_vc'),
                        inputs=[
                            tts_input, model_index, tts_speaker, pitch_adjust,
                            f0_method, feat_ratio
                        ],
                        outputs=[output_audio, output_msg, model_info],
                        fn=_example_edge_tts,
                        cache_examples=args.cache_examples,
                        run_on_click=args.cache_examples
                    )

    vc_convert_btn.click(
        vc_func,
        [
            input_audio, model_index, pitch_adjust, f0_method, feat_ratio,
            filter_radius, rms_mix_rate, resample_rate
        ],
        [output_audio, output_msg],
        api_name='audio_conversion'
    )

    tts_convert_btn.click(
        edge_tts_vc_func,
        [
            tts_input, model_index, tts_speaker, pitch_adjust, f0_method,
            feat_ratio, filter_radius, rms_mix_rate, resample_rate
        ],
        [output_audio, output_msg],
        api_name='tts_conversion'
    )

    full_song.click(fn=mix, inputs=[output_audio, as_audio_no_vocals], outputs=[new_song])

    model_index.change(
        update_model_info,
        inputs=[model_index],
        outputs=[model_info],
        show_progress=False,
        queue=False
    )
    
    gr.Markdown("### <center>注意❗：请不要生成会对个人以及组织造成侵害的内容，此程序仅供科研、学习及个人娱乐使用。</center>")
    gr.Markdown("### <center>🧸 - 如何使用此程序：填写视频网址和视频起止时间后，依次点击“提取声音文件吧”、“去除背景音吧”、“进行歌声转换吧！”、“加入歌曲伴奏吧！”四个按键即可。</center>")
    gr.HTML('''
        <div class="footer">
                    <p>🌊🏞️🎶 - 江水东流急，滔滔无尽声。 明·顾璘
                    </p>
        </div>
    ''')

app.queue(
    concurrency_count=1,
    max_size=20,
    api_open=args.api
).launch(show_error=True)
