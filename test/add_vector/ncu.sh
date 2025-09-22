n=1024 # 1 KB
iter=10

thres=$((1024 * 1024 * 256)) # 256 MB

n=$((1024 * 1024 * 512)) # 256 MB
thres=$((1024 * 1024 * 512)) # 256 MB

while [ $n -le $thres ]
do
    echo "################### Size: ${n} ###############################"
    for app in "gh"; do
        echo "------------------- App: ${app} ------------------------------"
        n_floats=$((n / 4))

        cmd="ncu --section C2CLink --replay-mode kernel -f -o ./profile/gh_${n} --section C2CLink ./bin/gh --n ${n_floats} --iter ${iter}"
        echo $cmd
        eval $cmd

        cmd="ncu --import ./profile/gh_${n}.ncu-rep --csv --log-file ./profile/gh_${n}.csv"
        echo $cmd
        eval $cmd

        echo ""
    done

    n=$((n * 4))
done