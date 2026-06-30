###############
Migration guide
###############

Basic workflow
==============

Call :py:meth:`~nvforest.load_model`, :py:meth:`~nvforest.load_from_sklearn`,
or :py:meth:`~nvforest.load_from_treelite_model`. Note that it is no longer
necessary to specify the ``is_classifier`` parameter.

.. code-block:: python

    # BEFORE
    fil_model = cuml.fil.ForestInference.load("xgb_model.ubj", is_classifier=True)
    fil_model.optimize(batch_size=1024)
    predictions = fil_model.predict(X_test)
    probabilities = fil_model.predict_proba(X_test)
    per_tree_pred = fil_model.predict_per_tree(X_test)
    lead_ids = fil_model.apply(X_test)

    # AFTER
    nvforest_model = nvforest.load_model("xgb_model.ubj")
    nvforest_model_optimized = nvforest_model.optimize(batch_size=1024)
    predictions = nvforest_model.predict(X_test)
    probabilities = nvforest_model.predict_proba(X_test)
    per_tree_pred = nvforest_model.predict_per_tree(X_test)
    lead_ids = nvforest_model.apply(X_test)

Device selection
================
Specify the ``device`` parameter when calling :py:meth:`~nvforest.load_model`.

.. code-block:: python

    # BEFORE
    with set_fil_device_type("cpu"):
        fil_model = cuml.fil.ForestInference.load("xgboost_model.ubj")
        result = fil_model.predict(data)

    # AFTER
    nvforest_model = nvforest.load_model("xgboost_model.ubj", device="cpu")

nvForest also differs from FIL when it comes to the behavior when no device is explicitly
specified. The ``device`` parameter defaults to ``"auto"``. nvForest will attempt to
load the tree model onto a GPU device, if one is available. If no GPU is available,
nvForest will fall back to the CPU.
