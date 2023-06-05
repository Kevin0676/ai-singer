# pull docker
FROM python:3.8.16-slim-bullseye

# install virtualenv
RUN apt update \
    && apt install -y aria2 wget curl tree unzip ffmpeg build-essential \
    && rm -rf /var/lib/apt/lists/*

# clean up
RUN apt-get clean; \
    rm -rf /etc/machine-id /var/lib/dbus/machine-id /var/lib/apt/lists/* /tmp/* /var/tmp/*; \
    find /var/log -name "*.log" -type f -delete

# set tmp
RUN mkdir -p /content/tmp
RUN chmod -R 777 /content/tmp
RUN rm -rf /tmp
RUN ln -s /content/tmp /tmp

# make dir
RUN mkdir -p /content
RUN chmod -R 777 /content

# try fix mplconfigdir
RUN mkdir -p /content/mplconfig
RUN chmod -R 777 /content/mplconfig

# try fix 
# RuntimeError: cannot cache function '__shear_dense': no locator available for file '/usr/local/lib/python3.8/site-packages/librosa/util/utils.py'
RUN mkdir -p /content/numbacache
RUN chmod -R 777 /content/numbacache

# try fix
# PermissionError: [Errno 13] Permission denied: '/.cache' (demucs)
RUN mkdir -p /content/demucscache
RUN chmod -R 777 /content/demucscache
RUN ln -s /content/demucscache /.cache

# set workdir
WORKDIR /content

# set environment
# PYTORCH_NO_CUDA_MEMORY_CACHING is can help users with even smaller RAM such as 2GB  (Demucs)
ENV PYTORCH_NO_CUDA_MEMORY_CACHING=1 \
    MPLCONFIGDIR=/content/mplconfig \
    NUMBA_CACHE_DIR=/content/numbacache

# upgrade pip
RUN python -m pip install --no-cache-dir --upgrade pip

# install library
RUN pip install --no-cache-dir --upgrade gradio
RUN pip install --no-cache-dir --upgrade setuptools wheel
RUN pip install --no-cache-dir faiss-gpu fairseq gradio ffmpeg ffmpeg-python praat-parselmouth pyworld numpy==1.23.5 numba==0.56.4 librosa==0.9.2

# copying requirements.txt
COPY requirements.txt /content/requirements.txt

# install requirements
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# copying files
COPY . .

# download hubert_base
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt -d /content -o hubert_base.pt

# download library infer_pack
RUN mkdir -p infer_pack
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://raw.githubusercontent.com/fumiama/Retrieval-based-Voice-Conversion-WebUI/main/infer_pack/attentions.py        -d /content/infer_pack -o attentions.py
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://raw.githubusercontent.com/fumiama/Retrieval-based-Voice-Conversion-WebUI/main/infer_pack/commons.py           -d /content/infer_pack -o commons.py
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://raw.githubusercontent.com/fumiama/Retrieval-based-Voice-Conversion-WebUI/main/infer_pack/models.py            -d /content/infer_pack -o models.py 
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://raw.githubusercontent.com/fumiama/Retrieval-based-Voice-Conversion-WebUI/main/infer_pack/models_onnx.py       -d /content/infer_pack -o models_onnx.py
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://raw.githubusercontent.com/fumiama/Retrieval-based-Voice-Conversion-WebUI/main/infer_pack/models_onnx_moess.py -d /content/infer_pack -o models_onnx_moess.py
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://raw.githubusercontent.com/fumiama/Retrieval-based-Voice-Conversion-WebUI/main/infer_pack/modules.py           -d /content/infer_pack -o modules.py
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://raw.githubusercontent.com/fumiama/Retrieval-based-Voice-Conversion-WebUI/main/infer_pack/transforms.py        -d /content/infer_pack -o transforms.py

# download library infer_pipeline.py
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/spaces/DJQmUKV/rvc-inference/raw/main/vc_infer_pipeline.py -d /content -o vc_infer_pipeline.py

# download library config.py and util.py
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/spaces/DJQmUKV/rvc-inference/raw/main/config.py -d /content -o config.py
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/spaces/DJQmUKV/rvc-inference/raw/main/util.py -d /content -o util.py

# check /tmp
RUN ls -l /tmp

# expose port gradio
EXPOSE 7860

# run app
CMD ["python", "app.py"]
