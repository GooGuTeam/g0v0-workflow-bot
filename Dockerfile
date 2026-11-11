FROM alpine:3.24

RUN apk add --no-cache \
    git \
    curl \
    ripgrep \
    coreutils \
    findutils \
    grep \
    sed \
    bash \
    jq \
    ca-certificates

RUN addgroup -S agent && \
    adduser -S -G agent agent

WORKDIR /workspace
RUN chown agent:agent /workspace && \
    install -d -m 1777 /run/pydantic-ai

USER agent

CMD ["/bin/sh"]
