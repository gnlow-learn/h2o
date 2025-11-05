import h2o
h2o.init()

model = h2o.load_model("dist/h2o/StackedEnsemble_AllModels_1_AutoML_1_20251105_135453")

import pandas as pd

test_df = pd.read_csv("data/test.csv")
test_h2o_frame = h2o.H2OFrame(test_df)

res = model.predict(test_h2o_frame)

res_df = res.as_data_frame().rename(
    columns={"predict": "Rented Bike Count"}
)

print(res_df.head())

res_df.to_csv(
    "dist/h2o.csv",
    encoding="utf-8",
    index_label="ID",
)
