from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="lamar",
    version="0.1.0",
    author="Ian Goodall-Halliwell",
    author_email="gooodallhalliwell@gmail.com",
    description="Label Augmented Modality Agnostic Registration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/LaMAR",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'lamar=lamar.cli:main'
        ],
    },
    include_package_data=True,
)