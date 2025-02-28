## Installation
Docker file
```
FROM a base image with vllm, pytorch and transformers, or use the one from RSD

# qwen math evaluation
RUN pip install --no-cache-dir \
       sympy \
       antlr4-python3-runtime==4.11.1 \
       word2number \
       Pebble \
       timeout-decorator

# parse answer
RUN pip install math-verify[antlr4_11.0]
```

## Run
check ./scripts