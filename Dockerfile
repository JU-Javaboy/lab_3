# Optional extension: package the inference service from serve_model.py
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
# TODO E3: install requirements without retaining pip's cache.
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 5000
# TODO E4: start the Flask app with `python serve_model.py`.
# Example: `python serve_model.py`
CMD ["python", "serve_model.py"]