.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_examples_decode_by_level.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_decode_by_level.py:


=============================
Decode by level
=============================

In this example, we load in some example data, and decode by level of higher order correlation.




.. code-block:: python

    # Code source: Lucy Owen
    # License: MIT

    # load timecorr and other packages
    import timecorr as tc
    import hypertools as hyp
    import numpy as np


    # load example data
    data = hyp.load('weights').get_data()

    # define your weights parameters
    width = 10
    laplace = {'name': 'Laplace', 'weights': tc.laplace_weights, 'params': {'scale': width}}

    # set your number of levels
    # if integer, returns decoding accuracy, error, and rank for specified level
    level = 2

    # run timecorr with specified functions for calculating correlations, as well as combining and reducing
    results = tc.timepoint_decoder(np.array(data), level=level, combine=tc.corrmean_combine,
                                   cfun=tc.isfc, rfun='eigenvector_centrality', weights_fun=laplace['weights'],
                                   weights_params=laplace['params'])

    # returns only decoding results for level 2
    print(results)

    # set your number of levels
    # if list or array of integers, returns decoding accuracy, error, and rank for all levels
    levels = np.arange(int(level) + 1)

    # run timecorr with specified functions for calculating correlations, as well as combining and reducing
    results = tc.timepoint_decoder(np.array(data), level=levels, combine=tc.corrmean_combine,
                                   cfun=tc.isfc, rfun='eigenvector_centrality', weights_fun=laplace['weights'],
                                   weights_params=laplace['params'])

    # returns decoding results for all levels up to level 2
    print(results)
**Total running time of the script:** ( 0 minutes  0.000 seconds)


.. _sphx_glr_download_auto_examples_decode_by_level.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: decode_by_level.py <decode_by_level.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: decode_by_level.ipynb <decode_by_level.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
