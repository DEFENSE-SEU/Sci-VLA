set -euo pipefail

log_root=${1:-}
if [ -z "$log_root" ]; then
    echo "Usage: $0 <log_root>"
    exit 1
fi
if [ ! -d "$log_root" ]; then
    echo "Directory $log_root does not exist."
    exit 1
fi

found_any=0

render_episode() {
    local episode_dir=$1
    echo "Rendering $episode_dir"
    python scripts/autobio_scripts/render.py "$episode_dir" --height 224 --width 224 --fps 50
    found_any=1
}

for dir in "$log_root"/*; do
    if [ ! -d "$dir" ]; then
        continue
    fi

    if [ -f "$dir/states.npy.zst" ]; then
        render_episode "$dir"
        continue
    fi

    for d in "$dir"/*; do
        if [ ! -d "$d" ]; then
            continue
        fi
        if [ -f "$d/states.npy.zst" ]; then
            render_episode "$d"
        fi
    done
done

if [ "$found_any" -eq 0 ]; then
    echo "No episode directories containing states.npy.zst found under $log_root."
    exit 1
fi
