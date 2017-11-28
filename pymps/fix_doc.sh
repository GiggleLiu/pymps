echo $1
sed -i "s/Parameters/Args/1" $1
sed -i "s/Return:/Returns:/1" $1
sed -i "s/:\([a-zA-Z0-9_]\+\): \(.\+\),/\1 (\2):/1" $1
