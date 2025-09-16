import gradio as gr


def extension__tts_generation_webui():
    ui_wrapper()
    return {
        "package_name": "extension_higgs_v2",
        "name": "Higgs V2 (Early Access)",
        "requirements": "git+https://github.com/rsxdalv/extension_higgs_v2@main",
        "description": "Higgs V2, Boson AI's TTS model",
        "extension_type": "interface",
        "extension_class": "text-to-speech",
        "author": "Boson AI",
        "extension_author": "rsxdalv",
        "license": "MIT",
        "website": "https://github.com/resemble-ai/higgs_v2",
        "extension_website": "https://github.com/rsxdalv/extension_higgs_v2",
        "extension_platform_version": "0.0.1",
    }


def ui_wrapper():
    from .gradio_app import ui

    ui()


if __name__ == "__main__":
    if "demo" in locals():
        locals()["demo"].close()
    with gr.Blocks() as demo:
        extension__tts_generation_webui()
    demo.launch()
