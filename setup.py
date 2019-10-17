from setuptools import setup, find_packages

setup(
    name='edmnets',
    version='0.0.1',
    author='Moritz Hoffmann, CMB Group Berlin',
    author_email='clonker[at]gmail.com',
    description='edmnets project',
    long_description='',
    packages=find_packages(),
    zip_safe=False,
    install_requires=['numpy', 'tensorflow', 'urllib3', 'tqdm']
)
