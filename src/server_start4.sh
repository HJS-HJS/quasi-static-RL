#!/bin/bash

PYTHON_FILE="train_test_server4.py"  # 실행할 Python 파일 이름
MAX_RETRIES=10                 # 최대 재실행 횟수
RETRY_COUNT=0                 # 현재 재시도 횟수

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    echo "[$((RETRY_COUNT+1))/$MAX_RETRIES] Python 스크립트를 실행합니다..."
    python3 "$PYTHON_FILE"
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "Python 스크립트가 정상적으로 종료되었습니다."
        break
    else
        echo "Python 스크립트가 비정상 종료되었습니다. 재실행합니다..."
        RETRY_COUNT=$((RETRY_COUNT + 1))
    fi
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "최대 재실행 횟수($MAX_RETRIES)에 도달했습니다. 스크립트를 종료합니다."
fi