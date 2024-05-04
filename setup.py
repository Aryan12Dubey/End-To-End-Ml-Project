from setuptools import find_packages,setup

from typing import List
def get_requirement(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if '-e .' in requirements:
            requirements.remove('-e .')

    return requirements




setup(
    name='End_To_End_ML_Project',
    version='0.0.1',
    author='Aryan',
    author_email='aryan.gyn12@gmail.com',
    packages=find_packages(),
    install_requires=get_requirement('requirements.txt')



)