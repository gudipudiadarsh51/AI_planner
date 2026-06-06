from setuptools import setup, find_packages

setup(
    name='sigr_trainer',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'pyarrow==12.0.1',
        'pandas==2.0.3',
        'numpy==1.24.3',
        'fsspec==2023.6.0',
        'gcsfs==2023.6.0',
        'google-cloud-storage==2.10.0',
    ]
)
