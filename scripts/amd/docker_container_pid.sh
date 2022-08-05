CONTAINER_ID=$1
ps aux | grep $CONTAINER_ID | awk '{print $1 $2}'
