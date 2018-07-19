# vernon

The “vernon” project is a toolkit for modeling synchrotron radio emission from
the magnetospheres of low-mass objects. It is written by Peter K. G. Williams
(<pwilliams@cfa.harvard.edu>).

## Requirements

- a C compiler to build the extension module that performs the
  [Summers (2005)](https://doi.org/10.1029/2005JA011159) coefficient
  computation.
- libraries and development files for [GSL](https://www.gnu.org/software/gsl/)
  (the GNU Scientific Library) version 2.4 or greater, for the Summers (2005)
  extension module.
- [numpy](https://www.numpy.org/) version 1.10 or greater.
- [pkgconfig](https://pypi.org/project/pkgconfig/) version 1.3 or greater.
- [pwkit](https://github.com/pkgw/pwkit/) version 0.8.19 or greater.
- [six](https://six.readthedocs.io/) version 1.10 or greater.

## Development

To build the extension modules so that you can type `import vernon` from a Python
interpreter running in this directory:

```
python setup.py build_ext --inplace
```

To set up shims so that typing `import vernon` in your Python interpreter will
load up whatever files you have here, without needing to re-install every time
you change something:

```
python setup.py develop
```

## Recent Changes

See [the changelog](CHANGELOG.md).

## Copyright and License

This code is copyright Peter K. G. Williams and collaborators. It is licensed
under the [MIT License](https://opensource.org/licenses/MIT).
