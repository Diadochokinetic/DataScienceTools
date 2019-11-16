from distutils.core import setup

setup(
    name='DataScienceTools',
    version='0.1dev',
    packages=['dstools','dstools.preprocessing', 'dstools.classifier'],
    license='MIT License',
    long_description=open('README.md').read(),
)