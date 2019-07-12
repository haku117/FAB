
src=score
rsync -avzhe 'ssh' --exclude '.*' haku7117@$1:/home/haku7117/results/$src/ /Users/guoyizhao/Documents/GitHub/results/$src/

rsync -avzhe 'ssh' --exclude '.*' haku7117@$1:/home/haku7117/fig/ /Users/guoyizhao/Documents/GitHub/results/fig/
