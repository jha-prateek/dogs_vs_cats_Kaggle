import numpy as np
from dc_model import model_name, model

img_size = 50

train_data = np.load("training.npy")

train = train_data[:-500]
test = train_data[-500:]


X = np.array([i[0] for i in train]).reshape(-1, img_size, img_size, 1)
Y = np.array([i[1] for i in train])

test_X = np.array([i[0] for i in test]).reshape(-1, img_size, img_size, 1)
test_Y = np.array([i[1] for i in test])

model.fit({"input": X}, {"targets": Y}, n_epoch=10,
          validation_set=({"input": test_X}, {"targets": test_Y}),
          snapshot_step=500, show_metric=True, run_id=model_name)

model.save(model_name)