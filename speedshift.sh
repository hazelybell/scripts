#!/bin/bash
if [[ "$(cat /sys/devices/system/cpu/intel_pstate/status)x" == "activex" ]]
then	true
else	echo "no speedshift"
	exit 2
fi
if [[ "$1x" == "powersavex" ]]
then	for i in /sys/devices/system/cpu/cpu*/cpufreq/energy_performance_preference
	do	echo "power" >>$i
	done
elif [[ "$1x" == "performancex" ]]
then	for i in /sys/devices/system/cpu/cpu*/cpufreq/energy_performance_preference
	do	echo "performance" >>$i
	done
else
	echo "Specify powersave or performance"
	exit 1
fi
for i in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
do	echo "powersave" >>$i
done
