if [ -z $PATH_TO_MODEL ]
then
  export PATH_TO_MODEL="model.pkl"
fi
if [ -z $PATH_TO_TRANSFORMER ]
then
    export PATH_TO_TRANSFORMER="transformer.pkl"
fi
if [[ ! -f $PATH_TO_MODEL ]]
then
    gdown  https://drive.google.com/uc?id=1BRLSyq4ODKJSLNaVhKkj-k6jFVh8-i46 --output=$PATH_TO_MODEL
else
    echo "Model exists"
fi

if [[ ! -f $PATH_TO_TRANSFORMER ]]
then
    gdown https://drive.google.com/uc?id=1QVQ7eJ-yiclsvmom7R6oEP0HheXea2DZ --output=$PATH_TO_TRANSFORMER
else
    echo "Transformer exists"
fi
python3 -m unittest test_main.py