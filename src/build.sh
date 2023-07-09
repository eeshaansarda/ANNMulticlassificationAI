# remove any compiled classes
rm *.class minet/*.class minet/*/*.class

# compile all .java files
echo "Compiling..."
javac -cp lib/*:minet:. minet/*.java minet/*/*.java *.java

