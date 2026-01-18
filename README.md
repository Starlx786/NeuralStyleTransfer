# NEURAL-STYLE-TRANSFER

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: SIMRANDEEP SINGH

*INTERN ID*: CTIS1238

*DOMAIN*: APP DEVELPOMENT 

*DURATION*: 4 WEEKS

*PROJECT DESCRIPTION*: 1. Introduction
Neural Style Transfer (NST) is a deep learning technique that enables the transformation of ordinary photographs into artistic images by combining the content of one image with the style of another. This project focuses on implementing a Neural Style Transfer model to apply artistic styles to photographs using a pre-trained convolutional neural network. The concept of Neural Style Transfer was first introduced by Gatys et al., demonstrating how deep neural networks can separate and recombine content and style from images. With the advancement of deep learning, NST has found applications in digital art, image editing, and creative media industries.
The project uses a pre-trained VGG19 model, which has been trained on the ImageNet dataset, to extract high-level features from images. By leveraging these features, the system generates a stylized image that preserves the structural content of the original photograph while adopting the artistic patterns of the style image.

2. Objective
The primary objective of this project is to implement a Neural Style Transfer system capable of applying artistic styles to photographs. The specific objectives are:
To understand the working of convolutional neural networks for feature extraction.
To apply a pre-trained deep learning model for artistic style transfer.
To generate a visually appealing stylized image by combining content and style.
To demonstrate the effectiveness of Neural Style Transfer using Python.

3. Methodology
The Neural Style Transfer system is implemented using Python and the PyTorch deep learning framework. A pre-trained VGG19 convolutional neural network is used as a feature extractor. The model is not trained from scratch; instead, its learned representations are utilized to extract content and style features from images.
The methodology involves the following steps:
Load and preprocess the content and style images.
Pass both images through the VGG19 network to extract feature maps.
Compute content loss, which ensures that the generated image retains the structure of the content image.
Compute style loss using the Gram matrix, which captures the texture and artistic patterns of the style image.
Combine content loss and style loss to form a total loss function.
Optimize the generated image using gradient descent to minimize the total loss.
The process is repeated for multiple iterations until a visually satisfactory stylized image is produced.

4. Tools and Technologies Used
Programming Language: Python
Framework: PyTorch
Model: Pre-trained VGG19
Libraries: Torchvision, Pillow, Matplotlib
Platform: Jupyter Notebook / Python Script

5. Results and Output
The system successfully generates stylized images by blending the content of a photograph with the artistic style of another image. The output images clearly demonstrate preserved structural details from the content image while incorporating textures, colors, and brushstroke patterns from the style image. The results confirm the effectiveness of Neural Style Transfer in creative image processing tasks.

6. Conclusion
This project successfully implements a Neural Style Transfer model using a pre-trained VGG19 network. The system demonstrates how deep learning techniques can be applied to artistic image transformation without training a model from scratch. Neural Style Transfer showcases the power of convolutional neural networks in understanding and manipulating visual content. The project can be further enhanced by optimizing runtime using fast style transfer models, adding a graphical user interface, or extending the system to support multiple artistic styles.

*OUTPUT*: 
