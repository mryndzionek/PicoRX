#!/usr/bin/env bash

wget https://github.com/raspberrypi/pico-sdk-tools/releases/download/v2.0.0-5/riscv-toolchain-14-x86_64-lin.tar.gz
sudo mkdir -p /opt/riscv/riscv-toolchain-14
sudo chown $USER /opt/riscv/riscv-toolchain-14
tar xvf riscv-toolchain-14-x86_64-lin.tar.gz -C /opt/riscv/riscv-toolchain-14
