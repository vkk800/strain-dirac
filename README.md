# strain-dirac
Calculates properties of superconducting or non-superconducting 2D Dirac electrons in an arbitrary quasi-1D vector potential

This is my personal code from a few years ago that calculates superconducting (or also non-superconducting, but that is
less interesting) properties of a 2D Dirac electron system under quasi-1D vector potential (e.g. a strain field). The code
was used to produce the results in this article: https://arxiv.org/abs/1601.04505 (Phys. Rev. B 93, 214505 (2016)).

Most of the code is contained in `diracp.py` and `tools.py`. The code was initially for my personal use, which means that it is
not extremely elegant and only some of it is documented. If you are interested in using or extending this code, you can ask
me about it; otherwise my motivation to improve it after this point is quite low.

Some examples on how to use the code are included:

* `example_spectrum.py`: Computes the normal state energy spectrum of a 2D Dirac Hamilonian in a periodic
potential of the type described in (https://www.nature.com/articles/nphys3109) or our article.

* `example_deltavsg.py`: Computes the self-consistent superconducting gap as a function of the coupling strength. The Tang-Fu type potential is used again.

There is also an incomplete C implementation of some subset of the functions in `dirac_c` folder. It was written to make the
code faster, but was not used in the end. It's mostly tested and should work, though.
