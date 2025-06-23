#!/usr/bin/env bash

PASSWORD="123456"

# 检查 redis 服务状态（exit 0=active, 非0=inactive）
echo "$PASSWORD" | sudo -S systemctl status redis > /dev/null 2>&1
if [ $? -eq 0 ]; then
  # 已经启动，无需操作
  exit 0
else
  # 未启动，则启动服务
  echo "$PASSWORD" | sudo -S systemctl start redis
fi
