for f in relaxed_pdb_tmp/*; do
    # [[ $f =~ ^(.{9}) ]]
    # dir=${BASH_REMATCH[1]}
    dir=${f:0:-4}
    mkdir -p "$dir" && mv "$f" "$dir"
done
