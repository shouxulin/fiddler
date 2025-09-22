n=1024 # 1 KB
iter=10

thres=$((1024 * 1024 * 256)) # 256 MB

while [ $n -le $thres ]
do
    echo "################### Size: ${n} ###############################"
    for app in "native" "gh"; do
        echo "------------------- App: ${app} ------------------------------"
        n_floats=$((n / 4))
        cmd="./bin/${app} --n ${n_floats} --iter ${iter}"
        
        echo $cmd
        eval $cmd
        echo ""
    done

    n=$((n * 4))
done