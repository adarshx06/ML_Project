from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT='e .'
def get_requirements(file_path: str) -> List[str]:
    """this function return the list of requirements from the file path"""

    requirements = []
    with open(file_path) as f:
        requirements = f.readlines()
        requirements = [req.replace("\n","") for req in requirements]
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements

setup(
    name='adarshmlproject',
    version='0.1.0',
    author='Adarsh',
    author_email='adarshthoke.us@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),

)
