from typing import List
from setuptools import find_packages, setup


def get_requirements(file_path: str) -> List:
    """_summary_

    Parameters
    ----------
    file_path : str
        Gets file path as parameter

    Returns
    -------
    List
        returns list of pakeges read from file
    """
    HYPHEN_E_DOT = "-e ."

    requirements = []
    with open(file_path) as file:
        requirements = file.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements


setup(
    name="ML_Project",
    version="0.0.1",
    author="HM Danyal Sajid",
    author_email="danyalsajid000@gmail.com",
    packages=find_packages(),
    requires=get_requirements("requirements.txt"),
)
