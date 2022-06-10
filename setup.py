from setuptools import setup

setup(
    name='cddd',
    version='1.2.2',
    packages=['cddd', 'cddd.data', 'cddd.data.default_model'],
    include_package_data=True,
    url='https://github.com/jrwnter/cddd',
    download_url='https://github.com/jrwnter/cddd/archive/refs/tags/1.2.2.tar.gz',
    license='MIT',
    author='Robin Winter',
    author_email='robin.winter@bayer.com',
    description='continous and data-driven molecular descriptors (CDDD)',
    entry_points={
        'console_scripts': [
            'cddd = cddd.run_cddd:main_wrapper',
        ],
    },
)
