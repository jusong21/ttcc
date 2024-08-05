
ls log/job.err* > err.txt

while read line
do 
	echo $line
	cat $line | tail -5
	echo " "
done < err.txt
