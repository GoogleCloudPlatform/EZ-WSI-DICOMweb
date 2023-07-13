# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
FROM python:3.11.3-slim
COPY *.* /
COPY WORKSPACE /
COPY LICENSE /
COPY test_docker/test_ez_wsi.sh /test_ez_wsi.sh
COPY ez_wsi_dicomweb /ez_wsi_dicomweb
RUN apt-get update && \
    apt-get install -y nano git && \
    apt install -y apt-transport-https curl gnupg -y && \
    curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor >bazel-archive-keyring.gpg && \
    mv bazel-archive-keyring.gpg /usr/share/keyrings && \
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/bazel-archive-keyring.gpg] https://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list && \
    apt update -y && apt install -y bazel python3-opencv && \
    apt full-upgrade -y && \
    apt install -y bazel-1.0.0 && \
    pip install --upgrade pip && \
    chmod +x /test_ez_wsi.sh
SHELL ["/bin/bash", "-c"]
RUN python3 -m venv requirements_env && \
    source requirements_env/bin/activate && \
    pip install -r requirements.txt
ENTRYPOINT /bin/bash