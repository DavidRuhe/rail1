from setuptools import setup

setup(
    name="rail1",
    install_requires=[
        "pytest",
        "pytest-cov",
    ],
    packages=["rail1"],
    version="0.0.1",
    author="",
    entry_points={
        "console_scripts": ["run=rail1.run.run:main", "devrun=rail1.run.devrun:main"],
    },
)
