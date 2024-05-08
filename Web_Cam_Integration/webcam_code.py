from tkinter import Image
import cv2
import torch
from torchvision import transforms
from model import FIVE_CNN

# Loading the trained model and setting it to evaluation mode
model = FIVE_CNN()
model.load_state_dict(torch.load('./vggnet_CE_adam.pth', map_location=torch.device('cpu')))  # Loading trained model weights onto CPU
model.eval()

# Define the mapping from class index to emotion
emotion_mapping = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Neutral',
    5: 'Sad',
    6: 'Surprise'
}


preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),  # Normalizing for grayscale image
])


# Initializing webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the captured frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Resize to match the model's expected input
    resized_gray = cv2.resize(gray, (48, 48))
    
    # Apply preprocessing transformations
    input_tensor = preprocess(resized_gray)
    input_tensor = input_tensor.unsqueeze(0)

    # Make predictions
    with torch.no_grad():
        predictions = model(input_tensor)
        predicted_class = predictions.argmax(dim=1).item()  # Get the predicted class index
        emotion = emotion_mapping[predicted_class]  # Map the predicted class index to the corresponding emotion

    # Display the emotion on the frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, emotion, (50, 50), font, 1, (0, 255, 0), 2, cv2.LINE_4)
    cv2.imshow('Webcam Feed', frame)
    
    # Break the loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
