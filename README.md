# Mango_Recognition
A final project about built a AI to recognize the disease/problems of the mango. (Updating)

Total five disease/problems:
  炭疽病 著色不佳 乳汁吸附 機械傷害 黑斑病

The Dataset belongs to "AI CUP 2020 Mango Image Recognition Challenge"
Therefore, we don't put dataset and the label file in github.

# Model
![alt text](https://github.com/davidnds98/Mango_recognition/blob/main/figure/model.png)

# Experiment
First experiment, we resize the image and then train the model and predict the performance.

![alt text](https://github.com/davidnds98/Mango_recognition/blob/main/figure/resize.png)

Second experiment, we resize the image and use the filter of opencv to make the data more sharp, and then train the model and predict the performance.

![alt text](https://github.com/davidnds98/Mango_recognition/blob/main/figure/resize%2Bsharp.png)

Third experiment, we resize the image and use the equalizeHist of opencv to equalize the data, and then train the model and predict the performance.

![alt text](https://github.com/davidnds98/Mango_recognition/blob/main/figure/resize%2Bequalize.png)

Last experiment, we mix all the method, and then train the model and predict the performance.

![alt text](https://github.com/davidnds98/Mango_recognition/blob/main/figure/resize%2Bsharp%2Bequalize.png)
