from setuptools import setup, find_packages

setup(
    name='stellarutil',
    version='0.0.11',
    author='Cameron Ross',
    author_email='ceross@cpp.edu',
    description= 'A utility package for CPP Fire Squad',
    packages=find_packages(),
    license='MIT',
    url='https://github.com/CPP-FIRE-Squad/stellarutil',
    # py_modules=['console', 'graph', 'simulation', 'calculations'],
    entry_points={
        'console_scripts': [
            'stellarutil = stellarutil.console:help',
            'stellarutil-create = stellarutil.console:create',
        ],
    },
    
)


# setup(
#     name='mypkg',
#     version='0.1',
#     packages=find_packages(),
#     package_data={'spam': ['data.txt']},
#     py_modules=['bacon']
# )

# root
# ├── spam
# │   ├── __init__.py
# │   ├── data.txt
# │   ├── eggs.py
# │   └── fizz
# │       ├── __init__.py
# │       └── buzz.py
# ├── bacon.py
# └── setup.py