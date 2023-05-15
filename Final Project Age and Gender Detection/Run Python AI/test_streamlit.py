from email.mime import image
import tensorflow as tf
from tensorflow import keras
from keras.utils import img_to_array, load_img
from keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2  
import cvlib as cv
import streamlit as st
from streamlit_option_menu import option_menu
import os
from os import listdir
import numpy as np
from numpy import asarray,save
import time

classes = ['man','woman']
age_classes = ['0-10','11-20','21-30','31-40','41-50','51-60','61-70','71-80','81-90','91-120']
age_classes_reversed = list(reversed(age_classes))

def main():
    st.set_page_config(
        page_title="Hello world",
        page_icon="chart_with_upwards_trend",
        layout="wide",
    )

    with st.sidebar:
        selected = option_menu("Main Menu", ["Home", "Author",'Upload Image','Camera Detection',"More", "Documentation"], 
            icons=['house', 'person-circle', 'cloud-arrow-up-fill','camera', 'bookmark-fill','folder-symlink'], menu_icon="cast", default_index=0)
        selected

    st.sidebar.markdown(
            """ Developed by Nguyen Phuc Dung   
                Email : dungduide2002@gmail.com  
                [Youtube] (https://www.youtube.com/channel/UC66F6NwHqdnCXYAmeu1qGXA)""")

    if selected == 'Home' :
        st.title(":green[Real Time Age And Gender Detection Application] :camera_with_flash:")
        html_temp_home1 = """<div style="background-color:#6D7B8D;padding:10px">
                                                <h4 style="color:white;text-align:center;">
                                                Age and Gender Detection application using OpenCV, Custom CNN model and Streamlit.</h4>
                                                </div>
                                                </br>"""
        st.markdown(html_temp_home1, unsafe_allow_html=True)
        st.write("""
                    The application has two functionalities.

                    1. Real time age detection using web cam feed.

                    2. Real time gender detection using web cam feed.

                    """)

    if selected == 'Author' :
        st.title(':green[INFORMATION] :card_index:')
        st.subheader("""
                    Developed by : Nguyen Phuc Dung """)
        st.subheader("""
                    Email : dungduide2002@gmail.com """)
        st.subheader("""
                    [Youtube] : (https://www.youtube.com/channel/UC66F6NwHqdnCXYAmeu1qGXA) """)
        st.subheader("""
                    [Facebook] : (https://www.facebook.com/phucdung.nguyen.56/) """)

    if selected == 'Upload Image' :
        # load model
        gender_model = load_model('gender_detection.model')
        age_model = load_model('Age_Detection.h5')

        st.title(':green[PLEASE UPLOAD YOUR PICTURE TO PREDICT] :frame_with_picture:')
        upload_file = st.file_uploader("Choose an image file", type=["jpg","jpeg","png"], label_visibility="collapsed")
        if not (upload_file is None) :
            st.write('**Size of file is :**',upload_file.size)
            img = image.load_img(upload_file, target_size = (350,350))
            st.image(img, channels="RGB")
            img_process = image.load_img(upload_file, target_size = (96,96))
            img_process=img_to_array(img_process)
            img_process=img_process.astype('float32')
            img_process=img_process/255
            img_process=np.expand_dims(img_process,axis=0)

            button_click = st.button("Age and Gender Predict")
            if button_click :
                progress_text = "Operation in progress. Please wait."
                my_bar = st.progress(0, text=progress_text)
                for percent_complete in range(100):
                    time.sleep(0.1)
                    my_bar.progress(percent_complete + 1, text=progress_text)
                gender = (gender_model.predict(img_process).argmax())
                age = (age_model.predict(img_process).argmax())

                gender_result = gender_model.predict(img_process).max()
                age_result = age_model.predict(img_process).max()

                st.subheader("Your Gender is : {} with Accuracy : {:.2f} %".format(classes[gender], gender_result*100))
                st.subheader("Your Age Range is : {} years old with Accuracy : {:.2f} %".format(age_classes_reversed[age], age_result*100))
                
    if selected == "Camera Detection" :
        # load model
        gender_model = load_model('gender_detection.model')
        age_model = load_model('Age_Detection.h5')
        st.title(':green[REAL TIME AGE AND GENDER DETECTION] :camera_with_flash:')
        st.subheader("Click on Start to use webcam and detect your age and gender real time :heart_decoration:")
        Camera_button = st.button("Start Camera")
        if Camera_button :
            # Camera frame on streamlit 
            frame_window = st.image([])
            webcam = cv2.VideoCapture(0)
            Stop_camera = st.button("Stop Camera")
            while webcam.isOpened():
                # read frame from webcam 
                status, frame = webcam.read()

                # apply face detection
                face, confidence = cv.detect_face(frame)

                # loop through detected faces
                for idx, f in enumerate(face):

                    # get corner points of face rectangle        
                    (startX, startY) = f[0], f[1]
                    (endX, endY) = f[2], f[3]

                    # draw rectangle over face
                    cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)

                    # crop the detected face region
                    face_crop = np.copy(frame[startY:endY,startX:endX])

                    if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
                        continue

                    # preprocessing for gender detection model
                    face_crop = cv2.resize(face_crop, (96,96))
                    face_crop = face_crop.astype("float") / 255.0
                    face_crop = img_to_array(face_crop)
                    face_crop = np.expand_dims(face_crop, axis=0)

                    # apply gender detection on face
                    conf = gender_model.predict(face_crop)[0] # model.predict return a 2D matrix, ex: [[9.9993384e-01 7.4850512e-05]]
                    age_conf = age_model.predict(face_crop)[0]

                    # get label with max accuracy
                    idx = np.argmax(conf)
                    age_idx = np.argmax(age_conf)

                    label = classes[idx]
                    age_label = list(reversed(age_classes))[age_idx]

                    label = "{}: {:.2f}%".format(label, conf[idx] * 100)
                    age_label = "{}: {:.2f}%".format(age_label, age_conf[age_idx] * 100)

                    Y = startY - 10 if startY - 10 > 10 else startY + 10

                    # write label and confidence above face rectangle
                    cv2.putText(frame, label, (startX, Y-16),  cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 0), 2)

                    cv2.putText(frame, age_label, (startX, Y+5),  cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 0), 2)

                    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    frame_window.image(imgRGB)

                if Stop_camera or cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    if selected == "More" :
        st.title(":green[Age & Gender Detection AI For Life] :robot_face:")
        st.header(":orange[General Introduction]")
        st.write("**:green[Age and gender detection] are the determination of the age and gender of the people in an image or video. This process allows essential data to be obtained and used very effectively in business development processes.**")
        st.subheader(":orange[How Does Age & Gender Detection AI Work?]")
        st.write("Artificial intelligence-based face analyzed technologies have gained serious importance. The most important factor underlying the popularity of these technologies is that they can be used very effectively in business life. Thanks to these technologies, such as customer segmentation, product development, and business diary have all become improvable. The most widely used artificial intelligence technologies based on face analysis are :green[age detection and face gender detection technologies]. These technologies provide highly effective online and physical store solutions, marketing methods, service improvement, and product development. Let's take a look at how these technologies work.")
        st.write("First of all, it should be noted that face :green[gender detection and age detection] are not easy tasks, and such tasks are complex even for us humans. While even two people of the same age and gender can look completely different from each other, it is a significant development that machines can successfully make these identifications. For example, long hair is an appearance trait usually associated with women, but this is not typically the case. Therefore, it is a complex process for technologies to make the correct detection.")
        st.write("The basis of :green[age detection and gender detection ai] in such applications is face detection technology. After artificial intelligence determines the faces in the image, it analyzes the gender and age of these faces and allows you to obtain statistics.")
        st.write("These are highly sophisticated algorithms and extremely difficult to use. In order to be able to perform such complex tasks and to make correct identifications, the algorithm needs to be trained with very large datasets. Today, Deep Learning algorithms are widely used for this, but companies such as Microsoft also develop software for these processes. The common side of this software is that it requires detailed knowledge, which means time and budget. On the other hand,web-based artificial intelligence solutions such as Cameraylze are at your side to meet the needs of your business. Cameralyze Age and Gender Solutions not only allows you to create your application from the no-code platform in minutes but also includes many third-party integrations to integrate it into your system.")
        st.subheader(":orange[Where is Age & Gender Detection AI Used?]")
        st.write(":green[Age detection and gender detection ai] have many different uses. For example, in-store and front-of-shop :green[age detection and gender detection] are very effective even in retail businesses. By using these technologies, businesses can identify customer segments, change or develop marketing strategies according to this segment, and change their product range.")
        st.write("For example, some software can detect the age range of users passing by a business and suggest advertisements that will affect this age range. It is possible to use age detection and gender detection not only for sales purposes but also for security purposes because algorithms can make more accurate age and gender determinations than humans.")
        st.write("The only negative aspect of these algorithms is that they have extremely complex work principles, as mentioned above, and they also require large data sets. This means a loss of both time and money. If you cannot find specialists in the field, it is challenging to learn all these technologies. In addition, software developed in this regard can be highly costly.")
        st.write("On the other hand, web-based artificial intelligence platforms such as Cameralyze allow you to improve your business without requiring high costs and intensive knowledge. You do not need any technology background to use Cameralyze; thanks to its user-friendly interface, completing all your work using only the :green[drag and drop] feature is possible.")
        st.subheader(":orange[How Can Businesses Use Age & Gender Detection AI for Their Benefits?]")
        st.write(":green[Age and gender detection] is essential for authentication, human-computer interaction, behavior analysis, product recommendation based on user preferences, and many other areas. Many companies needed age and gender data capture, but few solutions were available. Significant developments have been made in the past few years to meet this need. In the last decade, artificial intelligence classification systems have been used instead of manual classification systems for age and gender detection. With the introduction of artificial intelligence, the success rate of solving the problem has increased.")
        st.write(":green[Age detection]: Age detection can be used to place ads in the types of media most consumed by your target audience.")
        st.write(":green[Gender detection]: Gender detection can be used to determine whether a social media platform is more likely to show your product to men or women.")
        st.write("Therefore, why are :green[age and gender detection] so essential, and what advantages does it bring to companies? Companies use :green[demographic information] (age and gender information) to help them understand the characteristics of the people who buy their products and services. By detecting age and gender information, you will now be able to find out what kind of customers your brand appeals to. You can also use this information to attract new customers to your brand. With the right age and gender data, you can market more to your customers and stop spending to reach those who are not interested. In short, gender and age detection will make your brand stand out from your competitors and enable you to get more customers.")
        st.subheader(":orange[Companies may use age and gender detection for the following purposes:]")
        st.write("- Understand what products and services different customer groups want and can afford.")
        st.write("- Target marketing campaigns more precisely and thus reduce the cost per lead or sale..")
        st.write("- Identify how society is changing and how they need to adapt.")
        st.write("As you can see, :green[age and gender detection technologies] can be significant for your business. Using these technologies makes investing and developing your business at many different levels possible. If you want to take advantage of the possibilities of technology to make a suitable investment in your business, you can start using Cameralyze! Cameralyze is an AI-Based platform that you can perform :blue[face gender recognition online].")

    if selected == "Documentation" :
        st.title(":green[Documentation] :link:")
        st.subheader(":orange[Convolutional Neural Network Algorithm :]")
        st.write("**https://topdev.vn/blog/thuat-toan-cnn-convolutional-neural-network/**")
        st.write("**https://aptech.fpt.edu.vn/cnn-la-gi.html**")
        st.write("**https://viblo.asia/p/deep-learning-tim-hieu-ve-mang-tich-chap-cnn-maGK73bOKj2**")
        st.subheader(":orange[Link Youtube References :]")
        st.write("**https://www.youtube.com/watch?v=sTNMLLWnG1U**")
        st.write("**https://www.youtube.com/watch?v=JmvmUWIP2v8&t=638s**")
        st.write("**https://www.youtube.com/watch?v=Ebb4gUI2IpQ**")
        st.subheader(":orange[Icon Bootstrap :]")
        st.write("**https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app**")
        st.write("**https://icons.getbootstrap.com/?q=link**")
        st.subheader(":orange[Age and Gender Applications for Life]")
        st.write("**https://www.cameralyze.co/blog/age-gender-detection-top-use-cases**")


if __name__ == "__main__":
    main()