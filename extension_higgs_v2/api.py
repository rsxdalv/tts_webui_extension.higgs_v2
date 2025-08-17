import functools
import os
import tempfile
from typing import Optional

import gradio as gr
import numpy as np
import soundfile as sf
import torch

from tts_webui.utils.manage_model_state import (
    manage_model_state,
    rename_model,
    get_current_model,
)
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


def resolve_device(device: str):
    return get_best_device() if device == "auto" else device


def resolve_dtype(dtype: str):
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(dtype, torch.float32)


@manage_model_state("higgs_v2")
def get_engine(
    model_name: str = "just_a_placeholder",
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.float32,
):
    # Lazy import inside managed factory
    from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine

    # Engine doesn't use dtype directly, but we keep signature for consistency
    dev_str = device.type if isinstance(device, torch.device) else str(device)
    engine = HiggsAudioServeEngine(MODEL_PATH, AUDIO_TOKENIZER_PATH, device=dev_str)
    return engine


@manage_model_state("higgs_v2-whisper")
def get_whisper_model(
    model_name: str = "whisper-base",
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.float32,
):
    import whisper

    dev_str = device.type if isinstance(device, torch.device) else str(device)
    return whisper.load_model("base", device=dev_str)


def _build_messages(transcript: str, scene_description: Optional[str], ref_audio_path: Optional[str], ref_transcript: Optional[str]):
    from boson_multimodal.data_types import AudioContent, Message

    system_prompt = (
        f"Generate audio following instruction.\n\n<|scene_desc_start|>\n{scene_description or ''}\n<|scene_desc_end|>"
    )
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
    exaggeration=0.5,  # unused, kept for signature compatibility
    cfg_weight=0.5,  # unused
    temperature=0.8,
    audio_prompt_path=None,  # optional path to a reference voice wav
    # model
    model_name="just_a_placeholder",  # unused
    chunked=False,  # unused
    seed=-1,
    progress=gr.Progress(),
    scene_description: Optional[str] = None,
    # internal
    **kwargs,
):
    # text -> transcript
    transcript = text or ""
    if not transcript:
        raise gr.Error("Transcript (text) is empty.")

    device = get_best_device()
    progress(0.0, desc="Loading models…")
    engine = get_engine(
        model_name=generate_model_name(device, "float32"),
        device=torch.device(device),
        dtype=torch.float32,
    )

    ref_audio_path = None
    ref_transcript = None

    # If a reference audio is provided, transcribe it as the cloning text
    if audio_prompt_path:
        ref_audio_path = audio_prompt_path
        try:
            whisper_model = get_whisper_model(
                model_name="whisper-base",
                device=torch.device(device),
                dtype=torch.float32,
            )
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

    audio = output.audio.detach().cpu().numpy() if torch.is_tensor(output.audio) else output.audio
    sr = getattr(output, "sampling_rate", 22050)

    # Optional: write temp file if other parts need a path (not used in this extension flow)
    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(tmp.name, audio, sr)
    except Exception:
        pass

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Yield a single chunk compatible with extension expectations
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


@functools.wraps(_tts_generator)
def tts_stream(*args, **kwargs):
    try:
        # Single-yield streaming for compatibility
        yield from _tts_generator(
            *args, interrupt_flag=global_interrupt_flag, **kwargs
        )
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


# --- Minimal stubs for UI imports (no-op or basic behavior) ---
def move_model_to_device_and_dtype(device, dtype, cpu_offload=False):
    # Create/select a managed engine instance for the requested device/dtype
    try:
        resolved_device = resolve_device(device)
        if cpu_offload:
            resolved_device = "cpu"
        torch_device = torch.device(resolved_device)
        torch_dtype = resolve_dtype(dtype)
        get_engine(
            model_name=generate_model_name(resolved_device, dtype),
            device=torch_device,
            dtype=torch_dtype,
        )
        # Optionally ready whisper on same device
        get_whisper_model(
            model_name="whisper-base",
            device=torch_device,
            dtype=torch_dtype,
        )
        return True
    except Exception:
        return False


def compile_t3(model=None):
    return None


def remove_t3_compilation(model=None):
    return None


def vc(audio_in: str, audio_ref: str, progress=gr.Progress(), **kwargs):
    # Not supported for this engine; pass-through the input audio
    try:
        data, sr = sf.read(audio_in)
        return {"audio_out": (sr, data.astype(np.float32))}
    except Exception as e:
        raise gr.Error(f"VC unsupported: {e}")


async def interrupt():
    from .api import global_interrupt_flag

    global_interrupt_flag.interrupt()
    await global_interrupt_flag.join()
    return "Interrupt next chunk"
