
# run Part 1 with seed 123 and example.json
echo "------------- Running Part 1 with example.json ------------"
java -cp lib/*:minet:. P2Main data/Part1/train.txt data/Part1/test.txt 123 settings/example.json

# Running Part 1 with seed 123 and linear.json
echo "------------- Running Part 1 with linear.json ------------"
java -cp lib/*:minet:. P2Main data/Part1/train.txt data/Part1/test.txt 123 settings/linear.json

# Running Part 2 with seed 123 and the chosen hyper-parameter setting
echo "------------- Running Part 2 with Part2.json ------------"
java -cp lib/*:minet:. P2Main data/Part1/train.txt data/Part1/test.txt 123 settings/Part2.json

# Running Part 3 with seed 123 and without data preprocessing
echo "------------- Running Part 3 without data preprocssing ------------"
java -cp lib/*:minet:. P2Main data/Part3/train.txt data/Part3/test.txt 123 settings/Part2.json 0

# Running Part 3 with seed 123 and with data preprocessing
echo "------------- Running Part 3 with data preprocssing ------------"
java -cp lib/*:minet:. P2Main data/Part3/train.txt data/Part3/test.txt 123 settings/Part2.json 1

# Running Part 4 with seed 123
echo "------------- Running Part 4 ------------"
java -cp lib/*:minet:. P2MainPart4 data/Part1/train.txt data/Part1/test.txt 123 settings/Part2.json
