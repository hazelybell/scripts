#!/bin/bash
nvidia-settings -a '[gpu:0]/GPUGraphicsClockOffset[3]=-200'
nvidia-settings -a '[gpu:0]/GPUGraphicsClockOffset[2]=-200'
nvidia-settings -a '[gpu:0]/GPUMemoryTransferRateOffset[3]=-200'
nvidia-settings -a '[gpu:0]/GPUMemoryTransferRateOffset[2]=-200'
