#!/usr/bin/env python

from setuptools import find_packages, setup


setup(
    name="expert_remote_llm",
    version="v0.1.0",
    description="Remote LLM client for 'expert'",
    author="Liam Tengelis",
    author_email="liam.tengelis@blacktuskdata.com",
    packages=find_packages(),
    package_data={
        "": ["*.yaml"],
        "expert_remote_llm": [
            "py.typed",
        ],
    },
    install_requires=[
        "btdcore",
    ],
)
