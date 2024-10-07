import cv2
import numpy as np
from keras.models import model_from_json
from ytmusicapi import YTMusic
import webbrowser
import os
import tkinter as tk
from tkinter import *
from pathlib import Path

# Load emotion model
json_file = open('model/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)
emotion_model.load_weights("model/emotion_model.h5")
emotion_labels = ['happy', 'sad', 'fear', 'disgust', 'angry', 'neutral']

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"assets\frame0")


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)


prev_video_url = "" 

def search_music():
    global prev_video_url  
    music_name = entry_2.get()
    if music_name:
        ytmusic = YTMusic()
        search_results = ytmusic.search(music_name, filter="songs")
        if search_results:
            entry_1.delete(1.0, tk.END)  # Clear the existing text in entry_1
            for i, result in enumerate(search_results):
                song_title = result["title"]
                entry_1.insert(tk.END, f"{i + 1}. {song_title}\n")  
                if prev_video_url and result["videoId"] in prev_video_url:
                    entry_1.tag_add("pointer", f"{i + 1}.0", f"{i + 1}.end")
                    entry_1.tag_configure("pointer", background="#FF0000") 



def play_music():
    music_name = entry_2.get()
    if music_name:
        ytmusic = YTMusic()
        search_results = ytmusic.search(music_name, filter="songs")
        if search_results:
            video_urls = [f"https://www.youtube.com/watch?v={result['videoId']}" for result in search_results]
            webbrowser.open(video_urls[0])


def pause_music():
    os.system("taskkill /f /im chrome.exe")  # Change 'chrome.exe' to your browser's executable name


def detect_emotion():
    cap = cv2.VideoCapture(0)
    batch_size = 10  # Number of frames to process in each batch
    frames = []  # Batch of frames
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image from the webcam.")
            break

        # Preprocess the captured image
        resized_image = cv2.resize(frame, (48, 48))
        gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        input_data= np.expand_dims(gray_image, axis=-1)  # Add a channel dimension

        # Normalize the input data
        input_data = input_data.astype("float32") / 255.0

        # Add the frame to the batch
        frames.append(input_data)

        # Process the batch of frames
        if len(frames) == batch_size:
            batch = np.array(frames)

            # Pass the preprocessed frames through the emotion model for prediction
            predictions = emotion_model.predict(batch)

            for prediction in predictions:
                # Retrieve the predicted emotion
                emotion_index = np.argmax(prediction)
                predicted_emotion = emotion_labels[emotion_index]

                # Display the predicted emotion on the webcam
                cv2.putText(frame, predicted_emotion, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("Camera Feed", frame)

            # Reset the batch of frames
            frames = []

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and destroy the window
    cap.release()
    cv2.destroyAllWindows()
    recommend_music(predicted_emotion)


def recommend_music(emotion):
    ytmusic = YTMusic()
    search_results = ytmusic.search(emotion, filter="songs")
    if search_results:
        video_id = search_results[0]['videoId']
        video_url = f"https://www.youtube.com/watch?v={video_id}"
    if search_results:
            entry_1.delete(1.0, tk.END)  # Clear the existing text in entry_1
            for result in search_results:
                entry_1.insert(tk.END, result["title"] + "\n")
    webbrowser.open(video_url)



window = tk.Tk()
window.geometry("1360x720")
window.configure(bg="#E2CDD4")
canvas = tk.Canvas(
    window,
    bg="#E2CDD4",
    height=720,
    width=1360,
    bd=0,
    highlightthickness=0,
    relief="ridge"
)
canvas.place(x=0, y=0)
image_image_1 = PhotoImage(
    file=relative_to_assets("image_1.png"))
image_1 = canvas.create_image(
    680.0,
    360.0,
    image=image_image_1
)
canvas.create_text(
    446.0,
    34.0,
    anchor="nw",
    text="MUSIC PLAYER",
    fill="#000000",
    font=("Inter Light", 64 * -1)
)
canvas.create_text(
    163.0,
    184.0,
    anchor="nw",
    text="Now Playing",
    fill="#000000",
    font=("Inter Light", 64 * -1)
)
canvas.create_rectangle(
    22.0,
    133.0,
    1337.0,
    136.0,
    fill="#000000",
    outline="")
button_image_1 = PhotoImage(
    file=relative_to_assets("button_1.png"))
button_1 = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command=pause_music,
    relief="flat"
)
button_1.place(
    x=614.0,
    y=554.0,
    width=132.0,
    height=110.0
)
button_image_2 = PhotoImage(
    file=relative_to_assets("button_2.png"))
button_2 = Button(
    image=button_image_2,
    borderwidth=0,
    highlightthickness=0,
    command=play_music,
    relief="flat"
)
button_2.place(
    x=380.0,
    y=554.0,
    width=120.0,
    height=114.0
)
button_image_3 = PhotoImage(
    file=relative_to_assets("button_3.png"))
button_3 = Button(
    image=button_image_3,
    borderwidth=0,
    highlightthickness=0,
    command=lambda:[detect_emotion(), recommend_music()],
    relief="flat"
)
button_3.place(
    x=848.0,
    y=554.0,
    width=132.0,
    height=108.0
)
entry_image_1 = PhotoImage(
    file=relative_to_assets("entry_1.png"))
entry_bg_1 = canvas.create_image(
    425.0,
    383.5,
    image=entry_image_1
)
entry_1 = Text(
    bd=0,
    bg="#A2898D",
    fg="#000716",
    highlightthickness=0,font=16
)
entry_1.place(
    x=163.0,
    y=266.0,
    width=524.0,
    height=233.0
)

entry_image_2 = PhotoImage(
    file=relative_to_assets("entry_2.png"))
entry_bg_2 = canvas.create_image(
    1088.5,
    312.5,
    image=entry_image_2
)
entry_2 = Entry(
    bd=0,
    bg="#AD9FAB",
    fg="#000716",
    highlightthickness=0,font=22
)
entry_2.place(
    x=892.0,
    y=283.0,
    width=393.0,
    height=57.0
)
button_image_4 = PhotoImage(
    file=relative_to_assets("button_4.png"))
button_4 = Button(
    image=button_image_4,
    borderwidth=0,
    highlightthickness=0,
    command=search_music,
    relief="flat"
)
button_4.place(
    x=881.0,
    y=216.0,
    width=50.0,
    height=42.0
)
canvas.create_text(
    949.0,
    216.0,
    anchor="nw",
    text="Search",
    fill="#000000",
    font=("Inter Light", 40 * -1)
)
window.resizable(False, False)
window.mainloop()

