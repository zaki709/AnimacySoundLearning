# ベースイメージ
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

# 作業ディレクトリを設定
WORKDIR /app

# 必要なツールをインストール
RUN apt-get update && apt-get install -y \
    git \
    wget \
    libsndfile1 \
    vim-tiny \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Pythonのアップデート
RUN pip install --upgrade pip

# requirements.txtをコピーしてライブラリをインストール
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

# aliasをコンテナ内の.bashrcに書き込み
COPY .alias /root
RUN cat /root/.alias >> /root/.bashrc

# デフォルトのCMDを指定
CMD ["bash"]
