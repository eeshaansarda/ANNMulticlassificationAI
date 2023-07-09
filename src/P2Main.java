import java.io.FileReader;
import java.io.IOException;
import java.util.Random;

import org.jblas.DoubleMatrix;
import org.jblas.util.Logger;

import minet.Dataset;
import minet.layer.*;
import minet.loss.CrossEntropy;
import minet.loss.Loss;
import minet.optim.Optimizer;
import minet.optim.SGD;
import minet.util.Pair;

import org.json.simple.*;
import org.json.simple.parser.JSONParser;

public class P2Main {

    static Dataset trainset;
    static Dataset devset;
    static Dataset testset;

    static double[] meanTrainset;
    static double[] sdevTrainset;

    final static int NUMBER_OF_CLASSES = 3;

    public static void printUsage(){
        System.out.println("Input not recognised. Usage is:");
        System.out.println("\tjava -cp lib/jblas-1.2.5.jar:minet:. P2Main <train_data> <test_data> <random_seed> <json_setting_file> [<apply_preprocessing>]");
        System.out.println("\nExample 1: build an ANN for Part1 and Part2 (no preprocessing is applied)");
        System.out.println("\tjava -cp lib/jblas-1.2.5.jar:minet:. data/Part1/train.txt data/Part1/test.txt 123 settings/example.json");
        System.out.println("Example 2: build an ANN for Part3 (preprocessing is applied)");
        System.out.println("\tjava -cp lib/jblas-1.2.5.jar:minet:. data/Part3/train.txt data/Part3/test.txt 123 settings/example.json 1");
        System.out.println("Example 2: build an ANN for Part3 (preprocessing is not applied)");
        System.out.println("\tjava -cp lib/jblas-1.2.5.jar:minet:. data/Part3/train.txt data/Part3/test.txt 123 settings/example.json");
    }

    private static double[] mean(Dataset ds) {
        double averages[] = new double[ds.getInputDims()];
        double counter[] = new double[ds.getInputDims()];
        double x[][] = ds.getX();
        for(int i = 0; i < ds.getSize(); i++) {
            for(int j = 0; j < ds.getInputDims(); j++) {
                // Ignore 99999
                // What about calculating average of only values of that class?
                if(x[i][j] != 99999) {
                    averages[j] += x[i][j];
                    counter[j]++;
                }
            }
        }

        for(int i = 0; i < ds.getInputDims(); i++) {
            averages[i] /= counter[i];
        }

        return averages;
    }

    private static double[][] imputate(Dataset ds) {
        double x[][] = ds.getX();
        double averages[] = mean(ds);

        for(int i = 0; i < ds.getSize(); i++) {
            for(int j = 0; j < ds.getInputDims(); j++) {
                if(x[i][j] == 99999) {
                    x[i][j] = averages[j];
                }
            }
        }
        return x;
    }

    private static double[][] standardise(Dataset ds) {
        double x[][] = ds.getX();

        // z-score
        for(int i = 0; i < ds.getSize(); i++) {
            for(int j = 0; j < ds.getInputDims(); j++) {
                x[i][j] = (x[i][j] - meanTrainset[j]) / sdevTrainset[j];
            }
        }

        return x;
    }

    /**
     * apply data preprocessing (imputation of missing values and standardisation) on trainset (Part 3 only)
    */
    public static void preprocess_trainset(){
        //// YOUR CODE HERE (PART 3 ONLY)

        double y[][] = trainset.getY();
        Dataset ds = new Dataset(imputate(trainset), y);

        // Calculate mean
        meanTrainset = mean(trainset);

        // Standard deviation
        double x[][] = trainset.getX();
        sdevTrainset = new double[trainset.getInputDims()];
        for(int i = 0; i < trainset.getSize(); i++) {
            for(int j = 0; j < trainset.getInputDims(); j++) {
                sdevTrainset[j] += Math.pow((x[i][j] - meanTrainset[j]), 2);
            }
        }

        for(int i = 0; i < trainset.getInputDims(); i++) {
            sdevTrainset[i] /= trainset.getSize();
            sdevTrainset[i] = Math.sqrt(sdevTrainset[i]);
        }

        trainset = new Dataset(standardise(ds), y);
    }

    /**
     * apply data preprocessing (imputation of missing values and standardisation) on testset (Part 3 only)
    */
    public static void preprocess_testset(){
        //// YOUR CODE HERE (PART 3 ONLY)
        double y[][] = testset.getY();
        Dataset ds = new Dataset(imputate(testset), y);
        testset = new Dataset(standardise(ds), y);
    }

    private static void splitDataset(double amount) {
        double x[][] = trainset.getX();
        double y[][] = trainset.getY();

        int size = (int) (trainset.getSize() * amount);
        double devX[][] = new double[size][trainset.getInputDims()];
        double devY[][] = new double[size][1];
        for(int i = 0; i < size; i++) {
            for(int j = 0; j < trainset.getInputDims(); j++) {
                devX[i][j] = x[i][j];
            }
            devY[i][0] = y[i][0];
        }

        int trainSize = trainset.getSize() - size;
        double trainX[][] = new double[trainSize][trainset.getInputDims()];
        double trainY[][] = new double[trainSize][1];

        for(int i = 0; i < trainSize; i++) {
            for(int j = 0; j < trainset.getInputDims(); j++) {
                trainX[i][j] = x[i + size][j];
            }
            trainY[i][0] = y[i + size][0];
        }
        devset = new Dataset(devX, devY);
        trainset = new Dataset(trainX, trainY);
        // return null;
    }

    public static void main(String[] args){
        if (args.length < 4){
            printUsage();
            return;
        }

        // set jblas random seed (for reproducibility)
		org.jblas.util.Random.seed(Integer.parseInt(args[2]));
		Random rnd = new Random(Integer.parseInt(args[2]));

        // turn off jblas info messages
        Logger.getLogger().setLevel(Logger.WARNING);

		// load train and test data into trainset and testset
		System.out.println("Loading data...");
		//// YOUR CODE HERE
        try {
            trainset = Dataset.loadTxt(args[0]);
            testset = Dataset.loadTxt(args[1]);
        } catch (IOException e) {
            e.printStackTrace();
        }

        // check whether data-preprocessing is applied (Part 3)
        boolean preprocess = false;
        if (args.length==5){
            if (!args[4].equals("0") && !args[4].equals("1")){
                System.out.println("HERE" + args[4]);
                printUsage();
                System.out.println("\nError: <apply_preprocessing> must be either empty (off), or 0 (off) or 1 (on)");
                return;
            }
            if (args[4].equals("1"))
                 preprocess = true;
        }

        // apply data-processing on trainset
        if (preprocess)
            preprocess_trainset();

        // split train set into train set (trainset) and validation set, also called development set (devset)
        // suggested split ratio: 80/20
        trainset.shuffle(rnd); // shuffle the train data before we split. NOTE: this line was updated on Nov 11th.
        //// YOUR CODE HERE
        splitDataset(0.2); // 80% to trainSet and 20% to devSet


        // read all parameters from the provided json setting file (see settings/example.json for an example)
        //// YOUR CODE HERE
        JSONParser jsonParser = new JSONParser();
        JSONObject settings = null;
        try {
            settings = (JSONObject) jsonParser.parse(new FileReader(args[3]));
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(0);
        }

        // long numberOfHiddenLayers = (long)

		// build and train an ANN with the given data and parameters
        ANN ann = new ANN();
        //// YOUR CODE HERE
        Layer net = ann.build(
                  trainset.getInputDims(),
                  NUMBER_OF_CLASSES,
                  (int) (long) settings.get("n_hidden_layers"),
                  (int) (long) settings.get("n_nodes_per_hidden_layer"),
                  (String) settings.get("activation_function")
        );
        try {
            ann.train(
                    new CrossEntropy(),
                    new SGD(net, (double) settings.get("learning_rate")),
                    trainset,
                    devset,
                    (int) (long) settings.get("batchsize"),
                    (int) (long) settings.get("nEpochs"),
                    (int) (long) settings.get("patience"),
                    rnd);
        } catch (Exception e) {
            e.printStackTrace();
        }

        // evaluate the trained ANN on the test set and report results
        try{
            // apply data-preprocessing on testset
            if (preprocess)
                preprocess_testset();
		    double testAcc = ann.eval(testset);
		    System.out.println("accuracy on test set: " + testAcc);
        } catch(Exception e){
            System.out.println(e.getMessage());
            return;
        }
	}
}
