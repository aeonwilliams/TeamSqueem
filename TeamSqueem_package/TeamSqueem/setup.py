import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="TeamSqueem",
    version="0.0.1",
    author="Team Squeem: Aeon Williams, Ben Van Oostendorp, Alejandro Herrera",
    author_email="w.aeon@digipen.edu",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aeonwilliams/TeamSqueem",
    packages=setuptools.find_packages(),
    install_requires=[
        'Pillow', 'numpy', 'pandas', 'pathlib', 'matplotlib'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3',
)