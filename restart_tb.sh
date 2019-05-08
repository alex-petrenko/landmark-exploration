# sleep time, hours
sleep_t=0.002
times=0
filters=$2
port=$1
#Note: this doesn't pass the -h and -q flags

# while loop
while true
do
	python tb.py --port ${port} filters ${filters} &
	# tensorboard --port=${port} --logdir=${logdir} &
	last_pid=$!
	sleep ${sleep_t}h
	kill -9 ${last_pid}
	times=`expr ${times} + 1`
	echo "Restarted tensorboard ${times} times."
done
