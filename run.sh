#!/bin/bash
GREEN="\033[0;32m"
NC="\033[0m"

play="schiller-wilhelm-tell"

# check if fasttext model is there
# if not, then download it
mkdir -p fasttext_models
[[ ! -f fasttext_models/wiki.de.bin ]] && \
    echo -e "${GREEN}### downloading German fasttext model from fasttext.cc${NC}" && \
    cd fasttext_models && \
    wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.de.zip && \
    unzip wiki.de.zip && cd .. && \
    echo -e "${GREEN}### done downloading German fasttext model${NC}"

# check if the gerdracor example file is there
# if not, then download it
mkdir -p gerdracor
#[[ ! -f gerdracor/${play}.txt ]] && \
    echo -e "${GREEN}### downloading GerDraCor example text${NC}" && \
    cd gerdracor && \
    wget https://raw.githubusercontent.com/dracor-org/gerdracor/main/tei/${play}.xml && \
    cd .. && \
    python xmltotxt.py gerdracor/${play}.xml gerdracor/${play}.txt && \
    echo -e "${GREEN}### done downloading GerDraCor example text${NC}"


echo -e "${GREEN}### download german spacy model if not present${NC}"
#python -m spacy download de_core_news_lg
echo -e "${GREEN}### done downloading german spacy model${NC}"

# finally run the example
echo -e "${GREEN}### running the experiment${NC}"
mkdir -p processed
mkdir -p candidates
python chiasmus_example.py ${play}
echo -e "${GREEN}### done running the experiment${NC}"
