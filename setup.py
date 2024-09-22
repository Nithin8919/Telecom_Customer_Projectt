
from setuptools import find_packages, setup

setup(
    name='Telecomcustomer',
    version='0.0.1',
    author='Nithin',
    author_email='cherukumallinithin2003@gmail.com',
    description='A package for analyzing telecom customer data',
    # long_description=open('README.md').read(),  # Comment this out
    # long_description_content_type='text/markdown',  # Comment this out too
    url='https://github.com/yourusername/Telecomcustomer',  # Replace with your repository URL
    license='MIT',
    install_requires=[
        "scikit-learn",
        "pandas",
        "numpy"
    ],
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries',
    ],
)