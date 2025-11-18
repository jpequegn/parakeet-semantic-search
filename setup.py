from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="parakeet-semantic-search",
    version="0.1.0",
    author="Julien Pequegnot",
    description="Semantic search and recommendation engine for podcast transcripts",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/parakeet-semantic-search",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        "lancedb>=0.3.0",
        "sentence-transformers>=2.2.0",
        "numpy>=1.24.0",
        "duckdb>=0.9.0",
        "pandas>=2.0.0",
        "python-dotenv>=1.0.0",
        "click>=8.1.0",
        "pydantic>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
            "mypy>=1.0.0",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
        ],
        "streamlit": [
            "streamlit>=1.28.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "parakeet-search=parakeet_search.cli:main",
        ],
    },
)
