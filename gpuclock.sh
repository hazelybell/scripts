#!/bin/bash
nvidia-smi -pm 1
nvidia-settings -a '[gpu:0]/GPUGraphicsClockOffset[3]=0'
nvidia-settings -a '[gpu:0]/GPUGraphicsClockOffset[2]=0'
nvidia-settings -a '[gpu:0]/GPUMemoryTransferRateOffset[3]=-200'
nvidia-settings -a '[gpu:0]/GPUMemoryTransferRateOffset[2]=-200'
nvidia-smi -pl 300
