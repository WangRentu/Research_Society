#!/bin/bash
# Entrypoint for Cogito superimage
# Starts Jupyter Kernel Gateway for code execution

exec jupyter kernelgateway \
    --KernelGatewayApp.ip="${KG_IP:-0.0.0.0}" \
    --KernelGatewayApp.port=0 \
    --KernelGatewayApp.api='kernel_gateway.notebook_http' \
    --KernelGatewayApp.seed_uri='/dev/null'
