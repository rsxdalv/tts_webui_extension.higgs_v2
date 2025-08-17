
import setuptools
import re
import os

setuptools.setup(
	name="extension_higgs_v2",
    packages=setuptools.find_namespace_packages(),
    version="0.0.1",
	author="Boson AI",
	description="Higgs_v2 TTS extension for text-to-speech generation.",
	url="https://github.com/rsxdalv/extension_higgs_v2",
    project_urls={},
    scripts=[],
    install_requires=[
        "gradio",
        # "higgs_v2-tts"
        "higgs_v2-tts @ git+https://github.com/rsxdalv/higgs-audio@main",
        # "peft",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
