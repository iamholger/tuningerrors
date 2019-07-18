from setuptools import setup, find_packages
setup(
  name = 'tuningerrors',
  version = '0.1',
  description = 'Tuning errors',
  url = 'https://github.com/iamholger/tuningerrors',
  author = 'Andy Buckley, Holger Schulz',
  author_email = 'andy.buckley@cern.ch, hschulz@fnal.gov',
  packages = find_packages(),
  include_package_data = True,
  install_requires = [
    'numpy',
    'scipy',
    #'pyrapp',
    'mpi4py',
    'matplotlib'
  ],
  scripts=['bin/te-toy'],
  extras_require = {
  },
  entry_points = {
  },
  dependency_links = [
  ]
)
