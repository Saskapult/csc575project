cd clips

wget https://openslr.trmal.net/resources/12/dev-clean.tar.gz
unzip dev-clean.tar.gz

cd ..
uv run dataset.py librispeech
