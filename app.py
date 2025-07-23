import streamlit as st
import torch
from torch import nn

from torchvision import transforms
from PIL import Image

class TinyVGG(nn.Module):
  def __init__(self,
               input_shape:int,
               hidden_units:int,
               output_shape:int):
    super().__init__()
    self.conv_block_1=nn.Sequential(
        nn.Conv2d(in_channels=input_shape,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=0),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=0),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,
                     stride=2)
    )

    self.conv_block_2=nn.Sequential(
      nn.Conv2d(in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=0),
      nn.ReLU(),
      nn.Conv2d(in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=0),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2,
                    stride=2)
  )
    
    self.classifier=nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=hidden_units*13*13,
                  out_features=output_shape)
    )

  def forward(self,x):
    x=self.conv_block_1(x)
    # print(x.shape)
    x=self.conv_block_2(x)
    # print(x.shape)
    x=self.classifier(x)
    # print(x.shape)
    return x
  

model=TinyVGG(input_shape=3,
              hidden_units=10,
              output_shape=3)

model.load_state_dict(torch.load('baseline_vgg.pt'))
model.eval()

transform=transforms.Compose([
  transforms.Resize((64,64)),
  transforms.ToTensor()
])
class_dict={0:'BlackSpot', 1:'DowneyMildew', 2:'FreshLeaf'}

st.title("Leaf Disease Detection")
img=st.file_uploader("Upload an Image",type=['jpg','png'])

if img:
  image=Image.open(img)
  st.image(image, caption='Uploaded Image', use_column_width=True)

  img_tensor=transform(image).unsqueeze(0)

  with torch.inference_mode():
    output=model(img_tensor)
  pred_class=output.argmax(dim=1).item()

  label=class_dict[pred_class]
  if(label=='FreshLeaf'):
    st.write("The Leaf is healthy!")
  else:
    st.write(f'Predicted disease: {label}')