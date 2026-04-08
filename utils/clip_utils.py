"""
This file contains functions needed to run CLIP.

E6692 Spring 2026
"""

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def tokenize_and_generate_features(model, input_classes, tokenize_function, device = DEVICE):
    """
    Tokenize the text classes we defined
    and encode the tokens into text features
    
    params:
        model: CLIP model
        input_classes: List of defined classes
        tokenize_function: A function to tokenize our text classes
        device: computing device. cpu / cuda
    returns:
        text_features: encoded text features.
    """
    
    #####################################################################################
    # --------------------------- YOUR IMPLEMENTATION HERE ---------------------------- #
    #####################################################################################

#     raise Exception('utils.clip_utils.tokenize_and_generate_features not implemented!') # delete me

    # 1) Tokenize your classes using tokenize_function(input_classes)
    tokenized_text = tokenize_function(input_classes).to(device)
    
    # 2) Disable gradient computing
    with torch.no_grad():
        # 3) Generate text_features with model.encode_text(tokenized_text)
        text_features = model.encode_text(tokenized_text)
    
    #####################################################################################
    # --------------------------- END YOUR IMPLEMENTATION ----------------------------- #
    #####################################################################################
    
    return text_features
    

def clip_prediction(model, preprocess, image, text_features, device = DEVICE):
    """
    Perform zero-shot prediction using the CLIP framework.
    We will "manually" perform the predictions to demonstrate the conceptual ideas of CLIP.
    
    params:
        model: CLIP model
        preprocess: CLIP data preprocessor
        image: PIL.Image to predict for
        text_features: pre-computed text features.
        device: computing device. cpu / cuda
    returns:
        class_probabilities: Class probabilities
    """
    
    #####################################################################################
    # --------------------------- YOUR IMPLEMENTATION HERE ---------------------------- #
    #####################################################################################

#     raise Exception('utils.clip_utils.clip_prediction not implemented!') # delete me

    # 1) Preprocess the image. It's recommended that you first resize the image to be of size 256x256.
    # NOTE: as in previous labs, PyTorch models expect images to be batched regardless of how many samples you have.
    image = image.resize((256, 256))
    processed_image = preprocess(image).unsqueeze(0).to(device)
    # 2) Disable gradient computing
    with torch.no_grad():
        # 3) Generate image_features with model.encode_image(processed_image)
        image_features = model.encode_image(processed_image)
    
    # 4) Perform prediction using the model:
    # 4.1) Normalize the image features by dividing image_features by its norm (using .norm()) - Be mindful of the dimensions
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    # 4.2) Normalize the text features by dividing text_features by its norm (using .norm())  - Be mindful of the dimensions
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # 4.3) Compute prediction probabilities  by:
    # 4.3.1) computing the matrix product (dot product between vectors) between the image_features and text_features 
    mat_product = image_features @ text_features.T
    # 4.3.2) Multiplying the matrix product by 100
    mat_product = 100.0 * mat_product
    # 4.3.3) Apply SoftMax over the classes 
    class_probabilities = torch.softmax(mat_product, dim=-1)      
        
    # 5) Find the most probable class (index)
    most_probable_class = torch.argmax(class_probabilities, dim=-1).item()
        
    #####################################################################################
    # --------------------------- END YOUR IMPLEMENTATION ----------------------------- #
    #####################################################################################
    
    return class_probabilities, most_probable_class