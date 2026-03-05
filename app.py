import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Vehicle Counter AI",
    page_icon="🚗",
    layout="wide"
)

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>

.main-title{
font-size:40px;
font-weight:700;
color:#0e76a8;
}

.sub-title{
font-size:20px;
color:gray;
margin-bottom:20px;
}

.footer{
margin-top:50px;
text-align:center;
color:gray;
}

</style>
""", unsafe_allow_html=True)

# ------------------ HEADER ------------------
st.markdown('<p class="main-title">🚦 Hệ thống đếm phương tiện giao thông</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Ứng dụng YOLOv8 trong nhận diện và thống kê phương tiện</p>', unsafe_allow_html=True)

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8m.pt")

model = load_model()

# ------------------ SIDEBAR ------------------
st.sidebar.title("⚙️ Cấu hình hệ thống")

conf_threshold = st.sidebar.slider(
    "Độ tin cậy (Confidence)",
    0.0,1.0,0.5
)

selected_classes = [2,3,5,7]

st.sidebar.markdown("---")

st.sidebar.header("📌 Thông tin sinh viên")

st.sidebar.markdown("""
**Nguyễn Đình Trường**  
MSSV: **223332861**  

Lớp:  
*Kĩ thuật robot và trí tuệ nhân tạo - K63*
""")

# ------------------ UPLOAD VIDEO ------------------

uploaded_file = st.file_uploader(
    "📤 Tải video giao thông",
    type=["mp4","avi","mov"]
)

# ------------------ LAYOUT ------------------

col1,col2 = st.columns([3,1])

with col2:
    st.subheader("📊 Thống kê")

    car_metric = st.empty()
    bus_metric = st.empty()
    truck_metric = st.empty()
    motor_metric = st.empty()

with col1:
    frame_placeholder = st.empty()

# ------------------ PROCESS VIDEO ------------------

if uploaded_file is not None:

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)

    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = int(cap.get(5))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("output.mp4",fourcc,fps,(width,height))

    if st.button("🚀 Bắt đầu xử lý"):

        vehicle_count = {
            "car":0,
            "bus":0,
            "truck":0,
            "motorcycle":0
        }

        detected_ids = set()

        progress_bar = st.progress(0)

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        curr_frame = 0

        while cap.isOpened():

            ret,frame = cap.read()

            if not ret:
                break

            results = model.track(
                frame,
                persist=True,
                classes=selected_classes,
                conf=conf_threshold,
                tracker="botsort.yaml",
                verbose=False
            )

            if results[0].boxes.id is not None:

                boxes = results[0].boxes
                ids = boxes.id.cpu().numpy().astype(int)
                classes = boxes.cls.cpu().numpy().astype(int)

                for box,obj_id,cls_id in zip(boxes.xyxy,ids,classes):

                    label = model.names[cls_id]

                    x1,y1,x2,y2 = map(int,box)

                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,255),2)

                    cv2.putText(
                        frame,
                        f"{label} ID:{obj_id}",
                        (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0,255,255),
                        2
                    )

                    if obj_id not in detected_ids:
                        vehicle_count[label] = vehicle_count.get(label,0)+1
                        detected_ids.add(obj_id)

            # ------------------ UPDATE METRIC ------------------

            car_metric.metric("🚗 Car",vehicle_count["car"])
            bus_metric.metric("🚌 Bus",vehicle_count["bus"])
            truck_metric.metric("🚚 Truck",vehicle_count["truck"])
            motor_metric.metric("🏍️ Motorcycle",vehicle_count["motorcycle"])

            out.write(frame)

            curr_frame+=1
            progress_bar.progress(curr_frame/frame_count)

            preview = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

            if curr_frame % 5 == 0:
                frame_placeholder.image(preview,use_container_width=True)

        cap.release()
        out.release()

        st.success("✅ Hoàn thành xử lý video!")

        with open("output.mp4","rb") as f:
            st.download_button(
                "⬇️ Tải video kết quả",
                f,
                file_name="vehicle_count.mp4"
            )

# ------------------ FOOTER ------------------

st.markdown("""
<div class="footer">

Đề tài: **Ứng dụng AI trong nhận diện và đếm phương tiện giao thông**  
Sinh viên thực hiện: **Nguyễn Đình Trường - 223332861**

</div>
""",unsafe_allow_html=True)