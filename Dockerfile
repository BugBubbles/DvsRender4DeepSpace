FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libeigen3-dev \
    libopencv-dev \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . /workspace

RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_PHYSIC_EFFECTS=ON \
  -DENABLE_AER_DELAY=ON \
  && cmake --build build -j

CMD ["/bin/bash"]
