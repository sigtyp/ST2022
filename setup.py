from setuptools import setup, find_packages
import codecs

setup(
    name='sigtypst2022',
    version='1.4',
    license='MIT',
    description='Python Package for the Shared Task on Word Prediction',
    long_description=codecs.open('README.md', "r", "utf-8").read(),
    long_description_content_type='text/markdown',
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    author='Johann-Mattis List',
    author_email='mattis_list@eva.mpg.de',
    url='https://github.com/sigtyp/ST2022',
    keywords='word prediction',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    py_modules=["sigtypst2022"],
    include_package_data=True,
    zip_safe=False,
    platforms='any',
    python_requires='>=3.6',
    install_requires=["lingpy", "gitpython", "lingrex", "python-igraph"],
    entry_points={"console_scripts": ["st2022=sigtypst2022:main"]},
    extras_require={
        'dev': ['wheel', 'twine'],
        'test': [
            'pytest>=4.3',
            'pytest-cov',
            'coverage>=4.2',
        ],
    },
)
