TXTFILE=ptakopysk.txt

get_embs_cs:
	wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.cs.300.vec.gz

znaky:
	./embedinky_jako_ctverecky.py --DRAWLINES --BESTLINES=7 --EXP_FOR_OPACITY=0.5 --TXTFILE=ptakopysk.txt --BESTLINES_THRESHOLD=0.05

leftright:
	./embedinky_jako_ctverecky.py --LEFTRIGHT --TXTFILE=$(TXTFILE)

leftright_all:
	for f in *.txt; do ./embedinky_jako_ctverecky.py --no-show --LEFTRIGHT --TXTFILE=$$f; done

leftright_all_transcolor:
	for f in *.txt; do ./embedinky_jako_ctverecky.py --no-show --LEFTRIGHT --colors 13 --transcolor --EXP_FOR_OPACITY 0.5 --format svg --TXTFILE=$$f; done

