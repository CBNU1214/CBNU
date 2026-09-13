import streamlit as st
import cv2
import os 
import time
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer, WebRtcMode, RTCConfiguration 
import av 
import numpy as np
import base64 # 경고음 재생을 위해 추가

# 1. 'detector_logic.py' (수정본)에서 클래스를 가져옵니다.
try:
    # (사용자님의 'a classroom surveillance camera.py' 파일 이름을
    #  'detector_logic.py'로 변경했다고 가정합니다)
    from detector_logic import DrowsinessDetectorYOLOWorld
except ImportError:
    st.error("오류: 'detector_logic.py' 파일을 찾을 수 없습니다.")
    st.error("님의 원본 파이썬 파일('a classroom surveillance camera.py')의 이름을 'detector_logic.py'로 변경했는지 확인하세요.")
    st.stop()

# 2. Streamlit 페이지 설정
st.set_page_config(
    page_title="강의실 실시간 감지 시스템",
    layout="wide"
)

st.title("👨‍🏫 종합설계 | 강의실 실시간 감지 시스템")
st.markdown("---")

# 3. Streamlit 세션 상태에 AI 모델 로드 및 경고음 준비
if "detector" not in st.session_state:
    try:
        st.session_state.detector = DrowsinessDetectorYOLOWorld()
        print("Detector 인스턴스 생성됨")
    except Exception as e:
        st.session_state.detector = None
        st.error(f"🚨 AI 모델 로딩 실패: {e}")
        st.error("yolov8n-pose.pt와 yolov8s-world.pt 파일이 app.py와 같은 폴더에 있는지 확인하세요.")

if "run" not in st.session_state:
    st.session_state.run = False

# 경고음 재생 로직 추가 (앱 로드 시 1회만 실행)
if "beep_sound_base64" not in st.session_state:
    st.session_state.beep_sound_base64 = None
    if os.path.exists("beep.wav"):
        try:
            with open("beep.wav", "rb") as f:
                data = f.read()
                st.session_state.beep_sound_base64 = base64.b64encode(data).decode("utf-8")
                print("beep.wav 파일 로드 성공")
        except Exception as e:
            print(f"beep.wav 파일 로드 실패: {e}")
    else:
        st.warning("🔔 경고음 파일('beep.wav')을 찾을 수 없습니다. (app.py와 같은 폴더에 있어야 합니다)")

if "previous_alerts" not in st.session_state:
    st.session_state.previous_alerts = set() # 이전에 발생한 경고를 기억

RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# 4. 사이드바 (제어판)
with st.sidebar:
    st.header("🎛️ 제어판")
    st.markdown("---")
    
    start_disabled = (st.session_state.detector is None)
    
    if st.button("▶️ 프로그램 시작", type="primary", disabled=start_disabled):
        st.session_state.run = True
        st.session_state.previous_alerts = set() # 시작 시 경고 기록 초기화
        print("프로그램 시작됨: run=True")

    if st.button("⏹️ 프로그램 종료"):
        st.session_state.run = False
        if hasattr(st.session_state, "detector") and st.session_state.detector:
            # 프로그램 종료 시 활성 경고 목록을 정리 (선택 사항)
            st.session_state.detector.active_alerts.clear() 
        print("프로그램 종료(중지)됨: run=False")

    if start_disabled:
        st.error("모델 파일 로드 실패로 앱을 시작할 수 없습니다. 터미널을 확인하세요.")

    # --- 민감도 설정 (예시) 부분이 여기에서 삭제되었습니다 ---


# 5. st.session_state에서 detector 인스턴스를 *미리* 가져오기
if 'detector' in st.session_state:
    detector_instance_from_state = st.session_state.detector
else:
    detector_instance_from_state = None

# 6. 메인 화면 (영상 + 대시보드)
dashboard_col, main_video_col = st.columns([0.7, 0.3]) 

with main_video_col:
    st.header("📹 실시간 영상")
    
    # 6.1. 비디오 처리를 위한 클래스
    class VideoProcessor(VideoProcessorBase):
        def __init__(self, detector_instance):
            self.detector = detector_instance
            print("VideoProcessor: 초기화됨 (Detector 인스턴스 수신 완료)")

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            image = frame.to_ndarray(format="bgr24")
            
            # detector가 로드되지 않았거나, streamlit 세션 상태에 문제가 생겼을 경우 대비
            if not hasattr(self, 'detector') or self.detector is None:
                return av.VideoFrame.from_ndarray(image, format="bgr24")
            
            try:
                # detector_logic.py (원본 코드)의 process_frame 호출
                processed_image = self.detector.process_frame(image)
            except Exception as e:
                print(f"Error processing frame: {e}") 
                processed_image = image # 오류 발생 시 원본 이미지 반환
            
            return av.VideoFrame.from_ndarray(processed_image, format="bgr24")

    
    # 6.2. 비디오 스트리밍 실행
    if st.session_state.run:
        if detector_instance_from_state:
            webrtc_streamer(
                key="classroom_stream",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIGURATION,
                video_processor_factory=lambda: VideoProcessor(
                    detector_instance=detector_instance_from_state
                ),
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
                sendback_audio=False,
                desired_playing_state=True 
            )
        else:
            st.error("모델이 로드되지 않아 비디오 스트림을 시작할 수 없습니다.")
    else:
        if st.session_state.detector is None:
            st.error("모델 로딩에 실패했습니다. (파일 확인 필요)")
        else:
            st.info("'프로그램 시작' 버튼을 눌러주세요.")


with dashboard_col:
    st.header("📊 실시간 대시보드")

    # "실시간 경고 알림"을 위로 이동
    st.subheader("🚨 실시간 경고 알림")
    alert_placeholder = st.container(height=400) 

    # "감지된 학생"을 아래로 이동
    st.subheader("👨‍🎓 감지된 학생")
    student_list_placeholder = st.container(height=200) 


# 7. 기록 및 통계 (탭)
st.markdown("---")

tab_attendance, tab_escape = st.tabs([
    "✅ 출석자 기록", 
    "🏃 결석/이탈 기록"
])

# 7.1. 출석자 탭
with tab_attendance:
    st.header("🗂️ 출석 인정 기록")
    image_dir = 'temp_images'
    if os.path.exists(image_dir):
        # '_fixed'가 포함된 파일만 출석으로 간주 (원본 코드 로직 기준)
        image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') and '_fixed' in f]
        if not image_files:
            st.info("현재 출석으로 인정된 학생이 없습니다.")
        else:
            # 최신순 정렬
            image_files.sort(key=lambda x: os.path.getmtime(os.path.join(image_dir, x)), reverse=True)
            
            for img_file in image_files:
                file_path = os.path.join(image_dir, img_file)
                col1, col2 = st.columns([0.8, 0.2])
                with col1:
                    st.image(file_path, caption=img_file, use_column_width=True)
                with col2:
                    if st.button("🗑️ 삭제", key=f"del_att_{img_file}"):
                        try:
                            os.remove(file_path)
                            st.rerun() 
                        except Exception as e:
                            st.error(f"파일 삭제 오류: {e}")
    else:
        st.error("'temp_images' 폴더를 찾을 수 없습니다.")

# 7.2. 결석/이탈 탭
with tab_escape:
    st.header("🏃 결석/이탈 학생 기록")
    escape_dir = 'escape_images'
    if os.path.exists(escape_dir):
        escape_files = [f for f in os.listdir(escape_dir) if f.endswith('.jpg')]
        if not escape_files:
            st.info("기록된 이탈 학생이 없습니다.")
        else:
            # 최신순 정렬
            escape_files.sort(key=lambda x: os.path.getmtime(os.path.join(escape_dir, x)), reverse=True)
            
            for img_file in escape_files:
                file_path = os.path.join(escape_dir, img_file)
                col1, col2 = st.columns([0.8, 0.2])
                with col1:
                    st.image(file_path, caption=img_file, use_column_width=True)
                with col2:
                    if st.button("🗑️ 삭제", key=f"del_esc_{img_file}"):
                        try:
                            os.remove(file_path)
                            st.rerun() 
                        except Exception as e:
                            st.error(f"파일 삭제 오류: {e}")
    else:
        st.error("'escape_images' 폴더를 찾을 수 없습니다.")


# 8. 대시보드 실시간 업데이트
if st.session_state.run and detector_instance_from_state:
    try:
        # 학생 목록 업데이트
        with student_list_placeholder:
            fixed_students = [
                {"학생 ID": face_id, "상태": "고정됨"}
                for face_id, info in detector_instance_from_state.center_points.items() 
                if info.is_fixed
            ]
            if fixed_students:
                st.dataframe(fixed_students, use_container_width=True)
            else:
                st.info("감지된 학생이 없습니다.")

        
        # 경고음 재생 및 알림 표시 로직
        current_alert_keys = set(detector_instance_from_state.active_alerts.keys())
        previous_alert_keys = st.session_state.previous_alerts
        
        # 1. 새로 발생한 경고 찾기
        new_alerts = current_alert_keys - previous_alert_keys
        
        if new_alerts and st.session_state.beep_sound_base64:
            print(f"새로운 경고 감지, 경고음 재생: {new_alerts}")
            # 2. 경고음 1회 재생 (HTML/JS 자동 재생)
            st.components.v1.html(f"""
                <audio autoplay="true">
                  <source src="data:audio/wav;base64,{st.session_state.beep_sound_base64}" type="audio/wav">
                </audio>
                """, height=0)
        
        # 3. 다음 비교를 위해 현재 경고 상태 저장
        st.session_state.previous_alerts = current_alert_keys

        # 4. 활성 경고 알림창 업데이트 (기존 로직)
        with alert_placeholder:
            if detector_instance_from_state.active_alerts:
                # .items()로 딕셔너리의 복사본을 만들어 순회 (오류 방지)
                for alert_key, (message, img) in list(detector_instance_from_state.active_alerts.items()):
                    st.error(message) 
                    st.image(img, channels="BGR", use_column_width=True)
            else:
                 st.info("발생한 경고 알림이 없습니다.")

        # 0.5초마다 대시보드 새로고침
        time.sleep(0.5)
        st.rerun()

    except Exception as e:
        print(f"대시보드 업데이트 중 오류: {e}")
        st.session_state.run = False
        st.rerun()