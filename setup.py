from setuptools import setup, find_packages

setup(
    name="src",
    version="0.0.1",
    author="Aditya Y",
    author_email="yraditya895@gmail.com",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "scikit-learn",
        "pandas",
        "numpy",
        "google-api-python-client",
        "nltk",
        "python-dotenv",
        "from_root",
        "langdetect"
    ]
)