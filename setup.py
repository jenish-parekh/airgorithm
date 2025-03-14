from setuptools import setup, find_packages

# Lire le fichier requirements.txt pour récupérer les dépendances
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="AirGorithm",
    version="0.1.0",
    packages=find_packages(),
    install_requires=requirements,
    description="Aircraft body's damages detection",
    authors="Jenish Parekh, Stanislas Limouzi, Alexandre Cohen, Fayçal Radhi",
    author_email="jenish.p4rekh@gmail.com",
    url="https://github.com/jenish-parekh/airgorithm.git",
)
