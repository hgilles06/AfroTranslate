from setuptools import find_packages, setup
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name = "AfroTranslate",
    version = "0.0.5",
    author="Gilles HACHEME",
    author_email="gilles.hacheme@ai4innov.com",
    description = "This package allows you to obtain translations from Masakhane JoeyNMT based models. Masakhane is a grassroots research community aiming to revive and strengthen African languages through AI.",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/hgilles06/AfroTranslate",
    packages=find_packages(where='src'),
    package_dir={"": "src"},
    package_data={'': ['src/afrotranslate/links_models.csv']},
    include_package_data=True,
    use_scm_version=True,
    #cmdclass={
    #    "package": Package
    #},
    python_requires = ">=3.6",
    classifiers = [
    		'Development Status :: 3 - Alpha',
    		'License :: OSI Approved :: MIT License',
    		'Programming Language :: Python :: 3'
    		],
    install_requires = [
    			"joeynmt==1.3",
    			"googledrivedownloader==0.4",
    			"spacy==3.2.1"
    			]
)
