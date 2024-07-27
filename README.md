# Facial Emotion Recognition with VGG16 and SORT
We combine the trained VGG16 (on FER2013) with SORT (object tracking) to do emotion detection in the following way:

1. Use VGG16 to detect facial regions and emotions
2. Track the movement of the facial regions using SORT

Note that the accuracy obtained on the private_test_set is 72.5%. 

# Data and model checkpoint
1. Data *fer2013.csv*: https://drive.google.com/file/d/1IiNFgHVamdTyaspQYKp4-c8NFrU9LZkX/view?usp=drive_link
2. Checkpoint *epoch=87-step=19800.ckpt*:
   https://drive.google.com/file/d/13guzvXRbKxiIJzS1mEKClnr602qlww8l/view?usp=drive_link

# Youtube video
https://www.youtube.com/watch?v=f0vjMthxz0Q

