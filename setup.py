from setuptools import setup, find_packages

# Read the requirements from your requirements.txt
with open("requirements.txt", "r", encoding="utf-8") as f:
    required_packages = f.read().splitlines()

setup(
    name="smartcut",
    version="1.0.0",
    author="Santtu Keskinen",
    author_email="santtu.keskinen@gmail.com",
    description="CLI tool for cutting videos without recoding",
    url="https://github.com/skeskinen/smartcut",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        'Intended Audience :: End Users/Desktop',
    ],
    python_requires=">=3.11",
    install_requires=required_packages,
)
