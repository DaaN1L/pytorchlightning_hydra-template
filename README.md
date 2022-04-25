
# Broutonlab-python-template

Your project has to contain each of these files:
 ```
 .pre-commit-config.yaml
 poetry.lock
 pyproject.toml
 setup.cfg
```

Next, you have to install
- [pyenv](https://github.com/pyenv/pyenv)

- poetry, pre-commit
```sh
pip install poetry
poetry install
pre-commit install
```

And you need to set up the IDE to work with flake8/black (this repository has an example *.vscode* settings)
