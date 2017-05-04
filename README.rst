====
unwisest
====

.. image:: http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat
  :target: http://www.astropy.org
  :alt: Powered by Astropy Badge

.. image:: http://img.shields.io/badge/powered%20by-SciPy-orange.svg?style=flat
  :target: http://www.scipy.org
  :alt: Powered by Scipy Badge

.. image:: http://img.shields.io/badge/powered%20by-matplotlib-orange.svg?style=flat
  :target: http://www.matplotlib.org
  :alt: Powered by Matplotlib Badge


Description
-----------

This package is a point source confidence analyzer for WISE using unWISE.

Installation instructions and basic usage can be found on the `wiki <https://github.com/ctheissen/unwisest/wiki/>`_.


Full Documentation
------------------

Coming soon.

The easiest usage of this is to clone it into a directory.
You can use unwisest from the command line through the command:
    python unwisest ra dec
where "ra" is the right ascencion in decimal degrees and "dec" is the declination in decimal degrees. More funcationaliy can be accessed by import unwisest as a function within python.
  


History
-------

This program fits a 2-D Gaussian function to a point source in the unWISE tiles at a desired position. It uses empirical estimates based off ~1 million unWISE point sources to determine how "point-source-like" an object is in each WISE band. It also estimates the likelihood that the same point-source is extracted in each of the WISE bands. This should help to reduce contamination by chance alignments.


Project Status
--------------

Currently working on documentation.


License
-------

unwisest is free software licensed under an MIT-style license. For details see
the ``LICENSE.txt`` file.
