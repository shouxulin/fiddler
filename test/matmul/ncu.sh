batch_size=32
hidden_size=256
expert_Size=4096

n=1024 # 1 KB
iter=10

thres=$((1024 * 1024 * 256)) # 256 MB


n=$((1024 * 1024 * 512))
thres=$((1024 * 1024 * 512)) # 256 MB

while [ $n -le $thres ]
do
    echo "################### Size: ${n} ###############################"
    for app in "gh"; do
        echo "------------------- App: ${app} ------------------------------"
        n_floats=$((n / 2))
        expert_size=$((${n_floats} / ${hidden_size}))

        cmd="ncu --section C2CLink --replay-mode kernel -f -o ./profile/gh_${n} --section C2CLink ./bin/gh --batch ${batch_size} --hidden ${hidden_size} --expert ${expert_size} --iter ${iter}"
        echo $cmd
        eval $cmd

        cmd="ncu --import ./profile/gh_${n}.ncu-rep --csv --log-file ./profile/gh_${n}.csv"
        echo $cmd
        eval $cmd

        echo ""
    done

    n=$((n * 4))
done