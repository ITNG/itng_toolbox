import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
    name='itng',
    version='0.1',
    scripts=['itng'],
    author="Deepak Kumar",
    author_email="a.ziaeemehr@gmail.com",
    description="A Docker and AWS utility package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com:Ziaeemehr/itng_toolbox"
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
    ],
)
