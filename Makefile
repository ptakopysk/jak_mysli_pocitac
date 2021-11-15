
get_embs_cs:
	wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.cs.300.vec.gz

znaky:
	./embedinky_jako_ctverecky.py --DRAWLINES --BESTLINES=7 --EXP_FOR_OPACITY=0.5 --TXTFILE=ptakopysk.txt --BESTLINES_THRESHOLD=0.05

