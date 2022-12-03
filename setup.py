from setuptools import setup
from setuptools import find_packages

with open("README.md", "r") as f:
    long_description = f.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(name='gaia_oc_amd',
      version='0.1',
      description='Machine learning tool for the determination of new members of open clusters using Gaia data.',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://github.com/MGJvanGroeningen/gaia_oc_amd",
      author='Matthijs G. J. van Groeningen',
      author_email='matthijsvangroeningen@gmail.com',
      package_dir={'': './'},
      packages=find_packages('./'),
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
      ],
      python_requires='>=3.6',
      install_requires=requirements,
      package_data={'gaia_oc_amd.data_preparation': ['LogErrVsMagSpline.csv'],
                    'gaia_oc_amd.candidate_evaluation': ['pretrained_model/model_parameters',
                                                         'pretrained_model/hyper_parameters']},
      include_package_data=True)
