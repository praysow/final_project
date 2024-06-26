import os
import queue
import threading
import tempfile
from ultralytics import YOLO
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pyttsx3
import openai
import torch
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
import streamlit as st
from langchain_community.chat_models import ChatOpenAI

# OpenAI API 키 설정
openai.api_key = ''

# YOLO 모델 로드
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print(torch.__version__, 'device', DEVICE)

class TextToPrompt:
    def __init__(self):
        # 랭체인 LLM 설정
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            openai_api_key=openai.api_key
        )

        self.prompt_template = PromptTemplate(
            input_variables=["command"],
            template="사용자가 다음 명령을 내렸습니다: {command}. 어떤 클래스만 탐지해야 하나요?"
        )

        self.llm_chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt_template
        )

    def process_command(self, command):
        try:
            # LangChain을 사용하여 처리된 결과 받기
            response = self.llm_chain.run(command)
            response_text = response.strip().lower()  # 소문자로 변환하여 처리

            detected_classes = []

            # 특정 키워드 탐지
            keywords = ['빨간불', '초록불', '자전거', '킥보드', '라바콘', '횡단보도']

            for keyword in keywords:
                if keyword in response_text:
                    detected_classes.append(keyword)

            return detected_classes

        except Exception as e:
            print(f"Error processing command: {str(e)}")
            return []

# pyttsx3 초기화
engine = pyttsx3.init()
engine.setProperty('rate', 250)

# 큐 초기화
tts_queue = queue.Queue()

def tts_worker():
    while True:
        message = tts_queue.get()
        if message is None:
            break
        engine.say(message)
        engine.runAndWait()

# TTS 스레드 시작
tts_thread = threading.Thread(target=tts_worker)
tts_thread.start()

# YOLO 모델 로드
model = YOLO('best.pt')
model.model.names = ['빨간불', '초록불', '빨간불', '초록불', '자전거', '킥보드', '라바콘', '횡단보도']

# 클래스별 바운딩 박스 색상 지정
colors = {
    '빨간불': (255, 0, 0),  # 빨간색
    '초록불': (0, 255, 0),  # 초록색
    '자전거': (0, 0, 0),  # 검정색
    '킥보드': (128, 0, 128),  # 보라색
    '라바콘': (255, 165, 0),  # 주황색
    '횡단보도': (255, 255, 255)  # 횡단보도는 흰색으로 설정
}

# 한글 폰트 경로 설정
font_path = "C:/Windows/Fonts/malgun.ttf"
font = ImageFont.truetype(font_path, 20)

def detect_objects_in_video(video_path, desired_classes):
    cap = cv2.VideoCapture(video_path)
    frame_placeholder = st.empty()
    red_light_detected = False
    green_light_detected = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO를 사용하여 프레임에서 객체 탐지
        results = model.predict(frame, conf=0.5)

        detected_labels = set()
        tts_messages = set()

        for det in results:
            box = det.boxes
            for i in range(len(box.xyxy)):
                x1, y1, x2, y2 = box.xyxy[i].tolist()
                cls_id = int(box.cls[i])

                if cls_id < len(model.model.names):
                    label = model.model.names[cls_id]
                    if label not in desired_classes:
                        continue  # 원하는 클래스가 아닌 경우 무시
                    color = colors.get(label, (128, 128, 128))  # 기본 색상은 회색
                    print(f"Label: {label}, Color: {color}")  # 색상 값 확인
                else:
                    continue  # 원하는 클래스가 아닌 경우 무시

                conf = box.conf[i].item()
                detected_labels.add(label)

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color[::-1], 4)  # BGR 형식으로 변경

                img_pil = Image.fromarray(frame)
                draw = ImageDraw.Draw(img_pil)
                rgb_color = (color[2], color[1], color[0])  # BGR to RGB 변환 (색상 순서 뒤집기)
                draw.text((int(x1), int(y1) - 25), label, font=font, fill=rgb_color)
                frame = np.array(img_pil)

        # 감지된 라벨에 대해 TTS 메시지 추가
        if '빨간불' in detected_labels and '횡단보도' in detected_labels:
            tts_messages.add("빨간불이니 기다려 주세요")
            red_light_detected = True
            green_light_detected = False  # 초록불 감지 상태를 리셋
        elif '초록불' in detected_labels and '횡단보도' in detected_labels:
            tts_messages.add("초록불이니 길을 건너세요")
            green_light_detected = True
            red_light_detected = False  # 빨간불 감지 상태를 리셋
        else:
            red_light_detected = False
            green_light_detected = False

        # TTS 메시지를 큐에 추가 (매 프레임마다 추가되지 않도록)
        if red_light_detected or green_light_detected:
            for message in tts_messages:
                tts_queue.put(message)

        frame_placeholder.image(frame, channels="BGR")
        if cv2.waitKey(2) & 0xFF == ord('q'):
            break

    cap.release()

st.title("YOLO Object Detection App")

video_source = st.radio("비디오 소스 선택", ('캠코더', '비디오 파일'))

if video_source == '비디오 파일':
    video_file = st.file_uploader("비디오 파일 업로드", type=["mp4", "mov", "avi"])
    if video_file:
        with open("temp_video.mp4", "wb") as f:
            f.write(video_file.getbuffer())
        st.success("비디오 파일 업로드 완료")

user_prompt = st.text_input("탐지할 클래스를 쉼표로 구분하여 입력해주세요 (예: 횡단보도, 빨간불): ")

if st.button("탐지 시작"):
    if user_prompt:
        text_to_prompt = TextToPrompt()
        desired_classes = text_to_prompt.process_command(user_prompt)
        if desired_classes:
            st.write(f"탐지할 클래스: {desired_classes}")
            if video_source == '캠코더':
                detect_objects_in_video(0, desired_classes)  # 0은 웹캠을 의미
            else:
                detect_objects_in_video("temp_video.mp4", desired_classes)
        else:
            st.write("입력한 명령에서 관련 클래스를 찾을 수 없습니다.")
    else:
        st.write("탐지할 클래스를 입력해주세요.")

# 음성 출력 스레드 종료
tts_queue.put(None)
tts_thread.join()

print("Processing complete.")
