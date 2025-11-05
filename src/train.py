import pandas as pd

df = pd.read_csv("data/train.csv")

import h2o
from h2o.automl import H2OAutoML

h2o.init()
h2o_frame = h2o.H2OFrame(df)

y = "Rented Bike Count"
x = [col for col in h2o_frame.columns if col != y]

aml = H2OAutoML(
    max_models=20,
    max_runtime_secs=300,
    seed=42,
    include_algos=[
        "XGBoost",
        "GBM",
        "DRF",
        "GLM",
        "StackedEnsemble",
    ]
)

aml.train(x=x, y=y, training_frame=h2o_frame)

print(aml.leaderboard)

h2o.save_model(
    model=aml.leader,
    path="dist/h2o",
    force=True,
)
