source ./.conf
export $(cut -d= -f1 .conf)
logs="$(python test/test_yolov3.py 2>&1)"
status=$?

if [[ status -eq 0 ]]; then
  echo "Test successful."
  exit 0
else
  echo "Test failed." >&2
  echo $logs >&2
  exit 1
fi
