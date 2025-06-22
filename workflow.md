# Data Setting up

The data i got is not structured, of course it has different classes in different folders, but it is not split into train and test. So I use a bit of python code to split the data into train and test, then saved the zip and downloaded the zip file, now I'll be using the zip file.

# Model 1 -Tiny VGG

On 19th June 2025, i have used tinyVGG architecture, used in CNNExplainer website, for my use case of rose leaf disease prediction.

Again Trained on 22nd June- for 50 epochs with weight parameter for CELoss it converged with 79 percent test accuracy and 0.447 test_loss.

# Model 2- TinyMobileNet

On 21st June 2025, I have implemented MobileNetV1 architecture from scratch- now need to use a tiny version of this for my use case.

The tiny version of MobileNetV1- is replicated with initial 3 blocks and removed 5x blocks and again last 2 blocks so size increases from 64 to 1024. The test accuracy was low, model was poorly performing. So a full architecture of MobileNetV1 was used now even though the converged test accuracy is over 80 percent, the test_loss was more than tinyVGG.

This may be due to reduced parameter count- which reduces representational power- especially on small feature maps.

This makes sense: VGG has more parameters and stability, MobileNetV1 is light but needs careful tuning.
