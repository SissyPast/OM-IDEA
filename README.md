# `om-idea`

``om-idea`` is a framework for building aviation fleet-level analyses.
Such analyses can be used to understand the impact that system-level changes (e.g., introduction of more efficient aircraft) or factors external to aviation (e.g., fuel price increase) have at the level of an entire fleet of aircraft. 
It leverages the [OpenMDAO framework](https://openmdao.org/) to make it easier to create new analyses using existing blocks.

## Installation

1. Install the `uv` package manager using the [provided installation instructions](https://docs.astral.sh/uv/#installation). In a nutshell:

- For Windows: 

```
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

- For Linux and MacOS: 

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Clone this project into the directory of your choice:

```
git clone https://github.gatech.edu/rgautier6/om-idea
```

## Usage

Pick an example in the `examples` folder and run the `main.py` file using `uv`:

In Windows:

```
uv run python studies\formulations\energy_carriers_and_missions\main.py
```

In Linux:

```
uv run python studies/formulations/energy_carriers_and_missions/main.py
```

For more complex usage, please refer to the documentation.

## Documentation

Extensive documentation is being written and accessible in the `docs/` folder.
To access it, you can open `docs/_build/html/index.html` for a website version or [`docs/_build/pdf/latex/om-idea.pdf`](docs/_build/pdf/latex/om-idea.pdf) for a PDF version.
More information is available in the [docs' folder README](./docs/README.md).

## Support

To get help, please use Github's *Discussions* feature by navigating to the *Discussions* tab above:

![A screenshot of Github;s project tabs with the Discussions tab open](docs/source/_static/discussions.png)

Using this forum, the answer to your question will hopefully benefit other users!

If this doesn't work for you, feel free to reach out directly to Raphael via email or Teams ([raphael.gautier@gatech.edu](mailto:raphael.gautier@gatech.edu)).

## Roadmap

As of early 2026, the main effort is making it easier to use the tool by adding clearer interfaces to existing formulation, cleaning up the repository, and writing documentation.

See a [more detailed roadmap in the docs](./docs/source/docs/development/roadmap.rst), under `Development > Roadmap`.

## Contributing

If you would like to contribute, feel free to create a branch prefixed with your name and a short description of your changes, e.g., `raphael/updating-reference-data`.

## Authors and Acknowledgments

IDEA was originally created by Holger Pfaender. The `om-idea` rewrite using OpenMDAO and JAX was done by Raphael Gautier. 