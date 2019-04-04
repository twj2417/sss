from setuptools import setup, find_packages

setup(name='single scatter simulation',
      version='0.0.1',
      description='Algorithm to correct scatter',
      url='https://github.com/twj2417/sss',
      author='Weijie Tao',
      author_email='twj2417@gmail.com',
      license='Apache',
      packages=find_packages('src'),
      package_dir={'': 'src'},
      install_requires=[
          'click',
          'numpy',
          'numba',
          'h5py',
          'typing',
          'srfnef'
      ],
      entry_points="""
            [console_scripts]
            sss=sss.cli.main:sss
      """,
      zip_safe=False)
