source ./.conf
export $(cut -d= -f1 .conf)

if [[ $1 == "--help" ]] || [[ $1 == "-h" ]]; then
  echo "setup.sh [OPTION]"
  echo "Download weights and setup Yolo-V3 model."
  echo "Options:"
  echo "-f or --force   : Remove existing model and re-run setup."
  echo "-h or --help    : Display help message."
  exit
fi


if [[ $1 == "-f" ]] || [[ $1 == "--force" ]]; then
  echo "Removing saved weights."
  rm $modelweights
  rm $rawweights
fi

touch $buildlog
echo "Model build in progress. Do not remove this file" >> $buildlog

if [[ -f $modelweights ]]; then
  echo "Weights are found."
  echo "Testing setup..."
  sh test.sh 2>> $buildlog
  if [[ ! $? -eq 0 ]]; then
    echo "Test failed."
    echo "See $buildlog for details."
    exit 1
  fi
  echo "Model is ready."
  rm $buildlog
  exit 0
fi

if [[ ! -f $rawweights ]]; then
  echo "Downloading model weights..."
  curl $weightsurl -O >> $buildlog
  mv $raww $weightsdir/$raww >> $buildlog
  if [[ -f $rawweights ]]; then
    echo "Model weights successfully downloaded."
  else
    echo "Failed to fetch model weights."
    echo "See $buildlog for details."
    exit 1
  fi
else
  echo "Model weights already downloaded."
fi

if [[ ! -f $modelweights ]]; then
  echo "Setting up model..."
  python $buildscript 2> $buildlog
  if [[ ! -f $modelweights ]]; then
    echo "Failed to setup model"
    echo "See $buildlog for details."
    exit 1
  fi
fi


echo "Testing setup..."
sh test.sh 2>> $buildlog


if [[ ! $? -eq 0 ]]; then
  echo "Test failed."
  echo "See $buildlog for details."
  exit 1
fi

rm $buildlog
echo "Model is ready."
