mv $(du -d 1 logs | awk '$1 < 1024' | cut -f2) .desperated/logs # 清除大小小于1M的日志
