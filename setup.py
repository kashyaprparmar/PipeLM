from setuptools import setup, find_packages

setup(
    name="pipelm",
    version="0.1.0",
    description="An interface for terminal-based chatting with Hugging Face models.",
    author="Kashyap Parmar",
    author_email="kashyaprparmar@gmail.com",
    packages=find_packages(),
    py_modules=["pipelm", "app"],
    install_requires=[
        "rich>=10.0.0",
        "requests>=2.25.0",
        "huggingface-hub>=0.8.0",
        "python-dotenv>=0.19.0",
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "pydantic>=1.8.0",
        "transformers>=4.20.0",
        "torch>=2.0.0",
        "sentencepiece>=0.1.96",
    ],
    entry_points={
        "console_scripts": [
            "pipelm=pipelm:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
)