# Josiann: Just anOther SImulated ANNealing

Josiann is a Python library which implements the simulated annealing algorithm for noisy cost functions. It has
support for vectorized functions, multiprocessing and provides a parallel mode for optimizing several similar but
independent problems at once.

Source code (CeCILL-B): [https://github.com/Vidium/josiann](https://github.com/Vidium/josiann)

Documentation: [https://josiann.readthedocs.io/en/latest/index.html](https://josiann.readthedocs.io/en/latest/index.html)

## Installation
Josiann requires Python 3.8+ to run and relies on the following libraries :
- Numpy
- attrs
- tqdm

### Pip installation

```bash
pip install josiann
```

Optional diagnostic plots need the library `plotly` to be generated. You can install it alongside with Josiann :

```bash
pip install josiann[plot]
```

## Examples

Several examples of using Josiann can be found in the `docs/tutorials` folder.
