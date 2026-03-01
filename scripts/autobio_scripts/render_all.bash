log_root=$1
if [ -z "$log_root" ]; then
    echo "Usage: $0 <log_root>"
    exit 1
fi
if [ ! -d "$log_root" ]; then
    echo "Directory $log_root does not exist."
    exit 1
fi
for dir in $log_root/*; do
    if [ ! -d "$dir" ]; then
        continue
    fi
    for d in $dir/*; do
        if [ ! -d "$d" ]; then
            continue
        fi
        echo Rendering $d
        python scripts/autobio_scripts/render.py $d --height 224 --width 224 --fps 50
    done
done