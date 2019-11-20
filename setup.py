from setuptools import setup

setup(
    name='DataScienceTools',
    version='0.1dev',
    packages=['dstools',
            'dstools.preprocessing', 
            'dstools.classifier',
            'dstools.regressor',
            'dstools.wrapper'
            ],
    license='MIT License',
    long_description=open('README.md').read(),
    python_requires='>3.5',
    install_requires=[
                    'numpy>=1.16.2',
                    'pandas>=0.23.2',
                    'scikit-learn>=0.20.3',
                    'statsmodels>=0.9.0'
                    ]
)