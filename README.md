
# CKIP Dependency Parsing (Chinese)
DEMO Website: https://ckip.iis.sinica.edu.tw/service/dependency-parser/

# Model Download

Drive link: ###
```
mkdir demo_model
```

# Run CKIP tagger server

Download ```./data``` from https://github.com/ckiplab/ckiptagger (CkipTagger)
```Run in tmux```
```
cd tagger/

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

python3 server.py --port 5555
```

# Fast Parse

```
cd ..

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

sh fast_parse.sh
```

# fast_parse.sh

```
python3 parser.py --parse_sentence '今天天氣很好。' \
--port 5555 \
--gpu 1
```

# Output format --> List(Tuple)

(head:[index] - [token] [pos], tail:[index] - [token] [pos], [semantic relationship])

```
('4 - 好 VH', '1 - 今天 Nd', 'tmod')
('4 - 好 VH', '2 - 天氣 Na', 'nsubj')
('4 - 好 VH', '3 - 很 Dfa', 'advmod')
('0 - root root', '4 - 好 VH', 'root')
('4 - 好 VH', '5 - 。 PERIODCATEGORY', 'punct')
```