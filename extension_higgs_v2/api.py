import functools
import os
from typing import Optional

import gradio as gr
import numpy as np
import torch

from tts_webui.utils.manage_model_state import manage_model_state
from tts_webui.utils.get_path_from_root import get_path_from_root
from .InterruptionFlag import interruptible, InterruptionFlag

# Model identifiers
MODEL_PATH = "bosonai/higgs-audio-v2-generation-3B-base"
AUDIO_TOKENIZER_PATH = "bosonai/higgs-audio-v2-tokenizer"


def get_best_device():
    if torch.cuda.is_available():
        return "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def generate_model_name(device, dtype):
    return f"HiggsAudioEngine on {device} with {dtype}"


@manage_model_state("higgs_v2")
def get_engine(device: torch.device = torch.device("cuda")):
    from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine

    dev_str = device.type if isinstance(device, torch.device) else str(device)
    engine = HiggsAudioServeEngine(MODEL_PATH, AUDIO_TOKENIZER_PATH, device=dev_str)
    return engine


@manage_model_state("higgs_v2-whisper")
def get_whisper_model(device: torch.device = torch.device("cuda")):
    import whisper

    dev_str = device.type if isinstance(device, torch.device) else str(device)
    return whisper.load_model("base", device=dev_str)


def _build_messages(
    transcript: str,
    scene_description: Optional[str],
    ref_audio_path: Optional[str],
    ref_transcript: Optional[str],
):
    from boson_multimodal.data_types import AudioContent, Message

    system_prompt = f"Generate audio following instruction.\n\n<|scene_desc_start|>\n{scene_description or ''}\n<|scene_desc_end|>"
    messages = [Message(role="system", content=system_prompt)]
    if ref_audio_path and ref_transcript:
        messages.append(Message(role="user", content=ref_transcript))
        messages.append(
            Message(role="assistant", content=[AudioContent(audio_url=ref_audio_path)])
        )
    messages.append(Message(role="user", content=transcript))
    return messages


@interruptible
def _tts_generator(
    text,
    temperature=0.8,
    audio_prompt_path=None,  # optional path to a reference voice wav
    seed=-1,
    progress=gr.Progress(),
    scene_description: Optional[str] = None,
    **kwargs,
):
    # text -> transcript
    transcript = text or ""
    if not transcript:
        raise gr.Error("Transcript (text) is empty.")

    device = get_best_device()
    progress(0.0, desc="Loading models…")
    engine = get_engine(model_name=generate_model_name(device, "float32"))

    ref_audio_path = None
    ref_transcript = None

    # If a reference audio is provided, transcribe it as the cloning text
    if audio_prompt_path:
        ref_audio_path = audio_prompt_path
        try:
            whisper_model = get_whisper_model(model_name="whisper-base")
            result = whisper_model.transcribe(ref_audio_path)
            ref_transcript = result.get("text", "")
        except Exception:
            # If whisper fails, still try pure prompt cloning
            ref_transcript = ""

    # Seed
    if isinstance(seed, (int, float)) and int(seed) >= 0:
        torch.manual_seed(int(seed))

    # Messages
    progress(0.1, desc="Preparing prompt…")
    messages = _build_messages(
        transcript=transcript,
        scene_description=scene_description,
        ref_audio_path=ref_audio_path,
        ref_transcript=ref_transcript,
    )

    # Generate
    progress(0.2, desc="Generating audio…")
    from boson_multimodal.data_types import ChatMLSample

    output = engine.generate(
        chat_ml_sample=ChatMLSample(messages=messages),
        max_new_tokens=2048,
        temperature=float(temperature),
        top_p=0.95,
        top_k=50,
        stop_strings=["<|end_of_text|>", "<|eot_id|>"],
        ras_win_len=7,
        seed=int(seed) if isinstance(seed, (int, float)) and int(seed) >= 0 else None,
    )

    audio = (
        output.audio.detach().cpu().numpy()
        if torch.is_tensor(output.audio)
        else output.audio
    )
    sr = getattr(output, "sampling_rate", 22050)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    yield {
        "audio_out": (sr, audio.astype(np.float32)),
    }


global_interrupt_flag = InterruptionFlag()


@functools.wraps(_tts_generator)
def tts(*args, **kwargs):
    try:
        wavs = list(
            _tts_generator(*args, interrupt_flag=global_interrupt_flag, **kwargs)
        )
        if not wavs:
            raise gr.Error("No audio generated")
        full_wav = np.concatenate([x["audio_out"][1] for x in wavs], axis=0)
        return {
            "audio_out": (wavs[0]["audio_out"][0], full_wav),
        }
    except Exception as e:
        import traceback

        print(traceback.format_exc())
        raise gr.Error(f"Error: {e}")


def get_voices():
    voices_dir = get_path_from_root("voices", "higgs_v2")
    os.makedirs(voices_dir, exist_ok=True)
    results = [
        (x, os.path.join(voices_dir, x))
        for x in os.listdir(voices_dir)
        if x.lower().endswith(".wav")
    ]
    return results


async def interrupt():
    from .api import global_interrupt_flag

    global_interrupt_flag.interrupt()
    await global_interrupt_flag.join()
    return "Interrupt next chunk"
