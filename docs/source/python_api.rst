##########
Python API
##########

CUDA streams
============

GPU models use :external+cuda-python:py:class:`cuda.core.Stream` for model loading and
inference. Pass a stream with the ``stream`` parameter on any loading function, or omit it
to let nvForest create and retain a stream on the selected GPU. A supplied stream must be
associated with the same device as ``device_id``.

Model loading functions
=======================

.. role:: py(code)
   :language: python
   :class: highlight

.. autofunction:: nvforest.load_model

.. autofunction:: nvforest.load_from_sklearn

.. autofunction:: nvforest.load_from_treelite_model

.. autoclass:: nvforest.CPUForestInferenceClassifier
   :members:

.. autoclass:: nvforest.CPUForestInferenceRegressor
   :members:

.. autoclass:: nvforest.GPUForestInferenceClassifier
   :members:

.. autoclass:: nvforest.GPUForestInferenceRegressor
   :members:
