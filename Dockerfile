FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=7860 \
    MPLCONFIGDIR=/app/artifacts/.mplconfig

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . ./

EXPOSE 7860

CMD ["sh", "-c", "uvicorn digit_app.web:app --host ${HOST} --port ${PORT}"]