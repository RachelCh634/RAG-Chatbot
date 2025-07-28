FROM python:3.10-slim

# --- NETFREE CERT INSTALL ---
ADD https://netfree.link/dl/unix-ca.sh /home/netfree-unix-ca.sh 
RUN sh /home/netfree-unix-ca.sh
ENV NODE_EXTRA_CA_CERTS=/etc/ca-bundle.crt
ENV REQUESTS_CA_BUNDLE=/etc/ca-bundle.crt
ENV SSL_CERT_FILE=/etc/ca-bundle.crt
# --- END NETFREE CERT INSTALL ---

WORKDIR /app

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

RUN pip install --default-timeout=2000 --no-cache-dir \
    torch==2.0.1+cpu \
    torchvision==0.15.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --default-timeout=2000 --no-cache-dir open-clip-torch

COPY requirements.txt /app/

RUN pip install --default-timeout=2000 --no-cache-dir \
    --trusted-host pypi.org \
    --trusted-host files.pythonhosted.org \
    --trusted-host download.pytorch.org \
    -r requirements.txt

COPY . /app/

CMD ["sh", "-c", "python -B -m uvicorn main:app & streamlit run app.py"]