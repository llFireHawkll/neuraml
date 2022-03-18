from neuraml import __version__ as neuraml_version
from setuptools import setup, find_packages, Command

import os
import sys
from shutil import rmtree

here = os.path.abspath(os.path.dirname(__file__))

with open("./requirements.txt") as text_file:
    requirements = text_file.readlines()

requirements = list(map(lambda x: x.rstrip("\n"), requirements))
install_libraries = [x for x in requirements if not x.startswith("--extra-index")]


class UploadCommand(Command):
    """Support setup.py upload.
    python setup.py upload (command to upload the package to pypi)
    """

    description = "Build and publish the package."
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print("\033[1m{0}\033[0m".format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status("Removing previous builds…")
            rmtree(os.path.join(here, "dist"))
        except OSError:
            pass

        self.status("Building Source and Wheel (universal) distribution…")
        os.system("{0} setup.py sdist bdist_wheel --universal".format(sys.executable))

        self.status("Uploading the package to PyPI via Twine…")
        os.system("twine upload --skip-existing --verbose dist/* ")

        sys.exit()


setup(
    name="neuraml",
    version=neuraml_version,
    description="NeuraML, A Feature Packed ML Library For Data Processing and Model Training",
    url="https://github.com/llFireHawkll/neuraml",
    author="Sparsh Dutta",
    author_email="sparsh.dtt@gmail.com",
    python_requires=">=3.7.0",
    packages=find_packages(include=["neuraml", "neuraml.*"]),
    include_package_data=True,
    install_requires=install_libraries,
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    # $ setup.py publish support.
    cmdclass={
        "upload": UploadCommand,
    },
)
