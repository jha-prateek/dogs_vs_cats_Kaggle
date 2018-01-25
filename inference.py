import matplotlib.pyplot as plt
import numpy as np
from dc_model import model

img_size = 50
test_data = np.load("testing.npy")

fig = plt.figure()

for num, data in enumerate(test_data[48:60]):
    img_num = data[1]
    img_data = data[0]

    y = fig.add_subplot(3,4,num+1)
    orig = img_data
    plt_data = img_data.reshape(1,img_size,img_size,1)
    model_out = model.predict(plt_data)[0]

    if np.argmax(model_out) == 1:
        str_label = 'Dog'
    else:
        str_label = 'Cat'

    y.imshow(orig, cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()