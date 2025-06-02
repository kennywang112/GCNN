# 建立執行環境
FROM --platform=linux/amd64 python:3.9


# 更新Linux指令
RUN apt update
RUN apt-get install -y libgl1

# 建立工作資料夾，放置程式
WORKDIR /app


COPY requirements.txt /app/
COPY Download.py /app/
COPY app.py /app/
COPY models.py /app/
COPY templates /app/templates/
COPY mlflow /app/mlflow/
COPY model/face_landmarker.task /app/model/
COPY static /app/static/
COPY utils/ /app/utils/


RUN pip install --no-cache-dir -r requirements.txt
RUN python Download.py

EXPOSE 8080

ENTRYPOINT ["python", "app.py"]