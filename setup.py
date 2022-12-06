from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='gossips_cryptos',
      version="0.0.1",
      description="Predicting crypto price Model (api_pred)",
      license="MIT",
      author="Monica, Magnus, Mark, Babak",
      #author_email="contact@lewagon.org",
      url="https://github.com/monicasainer/gossips_cryptos",
      install_requires=requirements,
      packages=find_packages(),
      test_suite="tests",
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      zip_safe=False)
