import functools
import torch
import numpy as np
import gradio as gr
import os

from contextlib import contextmanager
from typing import TYPE_CHECKING

from tts_webui.utils.manage_model_state import (
    manage_model_state,
)
from tts_webui.utils.get_path_from_root import get_path_from_root

from .InterruptionFlag import interruptible, InterruptionFlag


if TYPE_CHECKING:
    from higgs_v2.tts import Higgs_v2TTS


@manage_model_state("higgs_v2")
def get_model(
    model_name="just_a_placeholder", device=torch.device("cuda"), dtype=torch.float32
):
    from higgs_v2.tts import Higgs_v2TTS

    model = Higgs_v2TTS.from_pretrained(device=device)
    # having everything on float32 increases performance
    return higgs_v2_tts_to(model, device, dtype)


@contextmanager
def higgs_v2_model(model_name, device="cuda", dtype=torch.float32):
    model = get_model(
        model_name=generate_model_name(device, dtype),
        device=torch.device(device),
        dtype=dtype,
    )

    yield model


@interruptible
def _tts_generator(
    text,
    exaggeration=0.5,
    cfg_weight=0.5,
    temperature=0.8,
    audio_prompt_path=None,
    # model
    model_name="just_a_placeholder",
    chunked=False,
    seed=-1,  # for signature compatibility
    progress=gr.Progress(),
    **kwargs,
):
    progress(0.0, desc="Retrieving model...")
    model = get_model(
        model_name=model_name,
        device=get_best_device(),
        dtype=torch.float32,
    )

    progress(0.1, desc="Generating audio...")

    return model.generate(
        text,
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
        temperature=temperature,
    )


global_interrupt_flag = InterruptionFlag()


@functools.wraps(_tts_generator)
def tts(*args, **kwargs):
    try:
        # Todo - Promise.all style parallel cascading for faster full audio generation (Omni) (very Important for slower GPUs)
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
        if x.endswith(".wav")
    ]
    return results


async def interrupt():
    from .api import global_interrupt_flag

    global_interrupt_flag.interrupt()
    await global_interrupt_flag.join()
    return "Interrupt next chunk"
