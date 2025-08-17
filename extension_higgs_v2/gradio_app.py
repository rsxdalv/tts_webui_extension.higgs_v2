import functools
import os
import tempfile
import gradio as gr

from tts_webui.decorators import *
from tts_webui.extensions_loader.decorator_extensions import (
    decorator_extension_inner,
    decorator_extension_outer,
)

from .api import (
    tts,
    get_engine,
    get_whisper_model,
)


@functools.wraps(tts)
@decorator_extension_outer
@decorator_apply_torch_seed
@decorator_save_metadata
@decorator_save_wav
@decorator_add_model_type("higgs_v2")
@decorator_add_base_filename
@decorator_add_date
@decorator_log_generation
@decorator_extension_inner
@log_function_time
def tts_decorated(*args, _type=None, **kwargs):
    return tts(*args, **kwargs)


def ui():
    higgs_v2_tts()


def higgs_v2_tts():
    """HiggsAudio Generation UI with Smart/Clone/Multi-voice features."""
    # Voice prompts directory (optional); if not found, preloaded_voices stays empty
    voice_prompts_dir = os.path.join(os.getcwd(), "examples", "voice_prompts")
    preloaded_voices = (
        [
            os.path.splitext(f)[0]
            for f in os.listdir(voice_prompts_dir)
            if f.lower().endswith(".wav")
        ]
        if os.path.exists(voice_prompts_dir)
        else []
    )

    def generate_audio(
        scene_description,
        transcript,
        voice_type,
        ref_audio_dropdown,
        custom_audio_upload,
        temperature,
        seed,
        speaker0,
        speaker0_custom_audio_upload,
        speaker1,
        speaker1_custom_audio_upload,
    ):
        import soundfile as sf
        import torch
        from boson_multimodal.data_types import AudioContent, ChatMLSample, Message

        # Validation for multi-speaker
        is_multi_speaker = "[SPEAKER" in (transcript or "")
        if is_multi_speaker and voice_type == "Voice Clone":
            return (
                gr.update(value=None),
                "For multi-speaker transcripts, please use 'Smart Voice' or 'Multi-voice Clone'.",
            )
        if not is_multi_speaker and voice_type == "Multi-voice Clone":
            return (
                gr.update(value=None),
                "For 'Multi-voice Clone', your transcript must include speaker tags like [SPEAKER0] and [SPEAKER1].",
            )

        # System prompt and messages
        system_prompt = f"Generate audio following instruction.\n\n<|scene_desc_start|>\n{scene_description}\n<|scene_desc_end|>"
        messages = [Message(role="system", content=system_prompt)]
        ras_win_len = 7

        device = "cuda" if torch.cuda.is_available() else "cpu"
        engine = get_engine(model_name=f"HiggsAudioEngine on {device} with float32")

        # Voice handling
        if voice_type == "Smart Voice":
            messages.append(Message(role="user", content=transcript))

        elif voice_type == "Voice Clone":
            # Single reference
            if ref_audio_dropdown == "Custom Upload":
                if custom_audio_upload is None:
                    return (
                        gr.update(value=None),
                        "Please upload a custom audio file (WAV format).",
                    )
                ref_audio_path = custom_audio_upload
                whisper_model = get_whisper_model(model_name="whisper-base")
                result = whisper_model.transcribe(ref_audio_path)
                ref_transcript = result.get("text", "")
            elif ref_audio_dropdown == "None" or not ref_audio_dropdown:
                ref_audio_path = None
                ref_transcript = None
            else:
                ref_audio_path = os.path.join(
                    voice_prompts_dir, f"{ref_audio_dropdown}.wav"
                )
                ref_transcript_path = os.path.join(
                    voice_prompts_dir, f"{ref_audio_dropdown}.txt"
                )
                if not os.path.exists(ref_transcript_path):
                    return (
                        gr.update(value=None),
                        f"Reference transcript not found at {ref_transcript_path}",
                    )
                with open(ref_transcript_path, encoding="utf-8") as f:
                    ref_transcript = f.read().strip()

            if ref_audio_path and ref_transcript is not None:
                messages.append(Message(role="user", content=ref_transcript))
                messages.append(
                    Message(
                        role="assistant",
                        content=[AudioContent(audio_url=ref_audio_path)],
                    )
                )
            messages.append(Message(role="user", content=transcript))

        elif voice_type == "Multi-voice Clone":
            if speaker0 == "None" or speaker1 == "None":
                return (
                    gr.update(value=None),
                    "Please select two speakers for multi-voice cloning.",
                )

            # Speaker 0
            if speaker0 == "Custom Upload":
                if speaker0_custom_audio_upload is None:
                    return (
                        gr.update(value=None),
                        "Please upload a custom audio file for Speaker 0 (WAV format).",
                    )
                ref_audio_path_0 = speaker0_custom_audio_upload
                whisper_model = get_whisper_model(model_name="whisper-base")
                result = whisper_model.transcribe(ref_audio_path_0)
                ref_transcript_0 = result.get("text", "")
            else:
                ref_audio_path_0 = os.path.join(voice_prompts_dir, f"{speaker0}.wav")
                ref_transcript_path_0 = os.path.join(
                    voice_prompts_dir, f"{speaker0}.txt"
                )
                if not os.path.exists(ref_transcript_path_0):
                    return (
                        gr.update(value=None),
                        f"Reference transcript not found for {speaker0}",
                    )
                with open(ref_transcript_path_0, encoding="utf-8") as f:
                    ref_transcript_0 = f.read().strip()

            # Speaker 1
            if speaker1 == "Custom Upload":
                if speaker1_custom_audio_upload is None:
                    return (
                        gr.update(value=None),
                        "Please upload a custom audio file for Speaker 1 (WAV format).",
                    )
                ref_audio_path_1 = speaker1_custom_audio_upload
                whisper_model = get_whisper_model(model_name="whisper-base")
                result = whisper_model.transcribe(ref_audio_path_1)
                ref_transcript_1 = result.get("text", "")
            else:
                ref_audio_path_1 = os.path.join(voice_prompts_dir, f"{speaker1}.wav")
                ref_transcript_path_1 = os.path.join(
                    voice_prompts_dir, f"{speaker1}.txt"
                )
                if not os.path.exists(ref_transcript_path_1):
                    return (
                        gr.update(value=None),
                        f"Reference transcript not found for {speaker1}",
                    )
                with open(ref_transcript_path_1, encoding="utf-8") as f:
                    ref_transcript_1 = f.read().strip()

            messages.extend(
                [
                    Message(role="user", content=f"[SPEAKER0] {ref_transcript_0}"),
                    Message(
                        role="assistant",
                        content=[AudioContent(audio_url=ref_audio_path_0)],
                    ),
                    Message(role="user", content=f"[SPEAKER1] {ref_transcript_1}"),
                    Message(
                        role="assistant",
                        content=[AudioContent(audio_url=ref_audio_path_1)],
                    ),
                    Message(role="user", content=transcript),
                ]
            )

        # Seed
        if seed is not None and str(seed).strip() != "":
            try:
                import torch as _t

                _t.manual_seed(int(seed))
            except Exception:
                pass

        # Generate
        from boson_multimodal.serve.serve_engine import HiggsAudioResponse  # type: ignore

        output: HiggsAudioResponse = engine.generate(
            chat_ml_sample=ChatMLSample(messages=messages),
            max_new_tokens=2048,
            temperature=float(temperature),
            top_p=0.95,
            top_k=50,
            stop_strings=["<|end_of_text|>", "<|eot_id|>"],
            ras_win_len=ras_win_len,
            seed=int(seed) if (seed is not None and str(seed).strip() != "") else None,
        )

        audio_data = (
            output.audio.detach().cpu().numpy()
            if hasattr(output, "audio")
            and getattr(output, "audio") is not None
            and hasattr(output.audio, "detach")
            else output.audio
        )
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(tmp_file.name, audio_data, output.sampling_rate)

        try:
            import torch as _t

            if _t.cuda.is_available():
                _t.cuda.empty_cache()
        except Exception:
            pass
        return tmp_file.name, None

    # UI layout
    with gr.Column():
        scene_description = gr.Textbox(
            label="Scene Description",
            value="Audio is recorded from a quiet room.",
            info="Describe the acoustic environment.",
        )
        transcript = gr.Textbox(
            label="Transcript",
            lines=5,
            placeholder="Enter text here. For multi-speaker, use [SPEAKER0], [SPEAKER1], etc.",
        )
        voice_type = gr.Radio(
            ["Smart Voice", "Voice Clone", "Multi-voice Clone"],
            label="Voice Type",
            value="Smart Voice",
            info=(
                "Smart Voice: Model selects voice. Voice Clone: Use a single reference audio. "
                "Multi-voice Clone: Use two reference audios for multi-speaker transcripts."
            ),
        )

        with gr.Group(visible=False) as voice_clone_group:
            ref_audio_dropdown = gr.Dropdown(
                choices=["None"] + preloaded_voices + ["Custom Upload"],
                label="Reference Audio",
                value="None",
                info="Select a pre-loaded voice or upload your own (used with Voice Clone).",
            )
            custom_audio_upload = gr.File(
                label=(
                    "Custom Reference Audio (Upload a WAV file if 'Custom Upload' is selected)"
                ),
                file_types=[".wav"],
            )

        with gr.Group(visible=False) as multi_voice_clone_group:
            speaker0_dropdown = gr.Dropdown(
                choices=["None"] + preloaded_voices + ["Custom Upload"],
                label="Speaker 0",
                value="None",
                info="Select a pre-loaded voice or upload your own for [SPEAKER0].",
            )
            speaker0_custom_audio_upload = gr.File(
                label=(
                    "Custom Reference Audio for Speaker 0 (Upload a WAV file if 'Custom Upload' is selected)"
                ),
                file_types=[".wav"],
                visible=False,
            )
            speaker1_dropdown = gr.Dropdown(
                choices=["None"] + preloaded_voices + ["Custom Upload"],
                label="Speaker 1",
                value="None",
                info="Select a pre-loaded voice or upload your own for [SPEAKER1].",
            )
            speaker1_custom_audio_upload = gr.File(
                label=(
                    "Custom Reference Audio for Speaker 1 (Upload a WAV file if 'Custom Upload' is selected)"
                ),
                file_types=[".wav"],
                visible=False,
            )

        temperature = gr.Slider(
            minimum=0.1,
            maximum=1.0,
            step=0.1,
            value=0.3,
            label="Temperature",
            info="Controls randomness (0.1 = less random, 1.0 = more random).",
        )
        seed = gr.Number(
            label="Seed",
            value=12345,
            info="Set for reproducible results. Leave blank for random.",
        )
        generate_btn = gr.Button("Generate", variant="primary")

    with gr.Column():
        output_audio = gr.Audio(
            label="Generated Audio", interactive=False, type="filepath"
        )
        error_md = gr.Markdown(visible=False)

    # Logic for toggling UI elements
    voice_type.change(
        fn=lambda value: (
            gr.update(visible=value == "Voice Clone"),
            gr.update(visible=value == "Multi-voice Clone"),
        ),
        inputs=voice_type,
        outputs=[voice_clone_group, multi_voice_clone_group],
    )

    def _toggle_custom(v):
        return gr.update(visible=v == "Custom Upload")

    ref_audio_dropdown.change(
        fn=_toggle_custom,
        inputs=ref_audio_dropdown,
        outputs=custom_audio_upload,
    )
    speaker0_dropdown.change(
        fn=_toggle_custom,
        inputs=speaker0_dropdown,
        outputs=speaker0_custom_audio_upload,
    )
    speaker1_dropdown.change(
        fn=_toggle_custom,
        inputs=speaker1_dropdown,
        outputs=speaker1_custom_audio_upload,
    )

    generate_btn.click(
        fn=generate_audio,
        inputs=[
            scene_description,
            transcript,
            voice_type,
            ref_audio_dropdown,
            custom_audio_upload,
            temperature,
            seed,
            speaker0_dropdown,
            speaker0_custom_audio_upload,
            speaker1_dropdown,
            speaker1_custom_audio_upload,
        ],
        outputs=[output_audio, error_md],
    )


if __name__ == "__main__":
    if "demo" in locals():
        locals()["demo"].close()
    with gr.Blocks() as demo:
        ui()

    demo.launch(
        server_port=7771,
    )
    # python -m workspace.extension_higgs_v2.extension_higgs_v2.gradio_app
