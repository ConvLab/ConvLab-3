'''
setup.py for ConvLab-3
'''
from setuptools import setup, find_packages

setup(
    name='convlab',
    version='3.0.2b',
    packages=find_packages(),
    license='Apache',
    description='An Open-source Dialog System Toolkit',
    long_description=open('README.md', encoding='UTF-8').read(),
    long_description_content_type="text/markdown",
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    setup_requires=['setuptools-git'],
    install_requires=[
        'joblib>=1.2.0',
        'pillow>=9.3.0',
        'protobuf>=3.20.2',
        'oauthlib>=3.2.1',
        'accelerate',
        'rouge-score',
        'sacrebleu',
        'tensorboardX',
        'boto3',
        'matplotlib',
        'seaborn',
        'tabulate',
        'python-Levenshtein',
        'requests',
        'numpy',
        'nltk',
        'scipy',
        'tensorboard',
        'torch>=1.10.1,<=1.13',
        'transformers>=4.17.0,<=4.24.0',
        'sentence-transformers>=2.2.2',
        'datasets>=2.0',
        'seqeval',
        'spacy',
        'simplejson',
        'unidecode',
        'jieba',
        'embeddings',
        'visdom',
        'quadprog',
        'fuzzywuzzy',
        'json_lines',
        'gtts',
        'pydub',
        'openai',
        'GitPython'
    ],
    extras_require={
        'develop': [
            "python-coveralls",
            "pytest-dependency",
            "pytest-mock",
            "requests-mock",
            "pytest",
            "pytest-cov",
            "checksumdir",
            "bs4",
            "lxml",
        ]
    },
    cmdclass={},
    entry_points={},
    include_package_data=True,
    url='https://github.com/ConvLab/ConvLab-3',
    author='convlab',
    author_email='convlab@googlegroups.com',
    python_requires='>=3.8',
    zip_safe=False
)
