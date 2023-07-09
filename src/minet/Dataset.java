// File: Dataset.java
// Dataset class
package minet;

import minet.util.Pair;
import org.jblas.DoubleMatrix;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;

/**
 * Dataset class for holding a set of (x, y) instances.
 * @author Phong Le
 */
public class Dataset {

    int currIndex;
    double[][] X;
    double[][] Y;

    public Dataset(double[][] X, double[][] Y) {
        this.X = X;
        this.Y = Y;
        this.currIndex = 0;
    }

    /**
     * Loading instances stored in a txt file.
     * <p>
     * The loaded txt file must have the following format:
     * <p>
     * First Line: <code>[number_of_instances] [x_dims] [y_dims]</code>  (if the task is classification, <code>y_dims</code> is always 1)
     * regardless to the number of categories.
     * <p>
     * Each following line: [<code>x_dims</code> float numbers seperated by a space] ; [<code>y_dims</code> float numbers seperated by a space]     
     * 
     * @param path a string, the path of the txt file.
     * @return a Dataset
     * @throws IOException
     */
    public static Dataset loadTxt(String path) throws IOException {
        // first line: [number of samples] [xDims] [yDims]
        // each line of file : [entries of X] ; [entries of Y]
        BufferedReader br = new BufferedReader(new FileReader(path));

        // first line
        String[] ss = br.readLine().split(" ");
        int size = Integer.valueOf(ss[0]);
        int xDims = Integer.valueOf(ss[1]);
        int yDims = Integer.valueOf(ss[2]);

        double[][] X = new double[size][xDims];
        double[][] Y = new double[size][yDims];

        for (int i = 0; i < size; i++) {
            ss = br.readLine().split(" ; ");
            String[] sx = ss[0].split(" ");
            String[] sy = ss[1].split(" ");
            for (int j = 0; j < sx.length; j++) {
                X[i][j] = Double.valueOf(sx[j]);
            }
            for (int j = 0; j < sy.length; j++) {
                Y[i][j] = Double.valueOf(sy[j]);
            }
        }

        return new Dataset(X, Y);
    }


    /**
     * Must be called before each epoch to reset the minibatch iteration.
     */
    public void reset() {
        this.currIndex = 0;
    }

    /**
     * Get the number of the instances stored.
     */
    public int getSize() {
        return X.length;
    }

    /**
     * Get the number of input features
     */
    public int getInputDims() {
        return X[0].length;
    }

    /**
     * Get the number of the output variables 
     * <p>
     * NOTE: 
     *      Do not mistaken this with outdims in an ANN for classification task, they are NOT the same!
     *      For classification tasks in our module, this will always return 1. For classification tasks where this number is larger than 1, see multi-label classification.
     */
    public int getOutDims() {
        return Y[0].length;
    }    

    /**
     * Shuffle the ordering of instances in the dataset (given a random generator, so the behaviour is reproducible). 
     * Should be called before each epoch during training.
     * @param rnd a random generator
     */
    public void shuffle(Random rnd) {               
        for (int i = this.getSize() - 1; i > 0; i--)
        {
            int index = rnd.nextInt(i + 1);
            // Simple swap
            double[] tmp = X[i];
            X[i] = X[index];
            X[index] = tmp;

            tmp = Y[i];
            Y[i] = Y[index];
            Y[index] = tmp;
        }
        this.currIndex = 0;
    }

    /**
     * Get all instances (feature values and labels) at once.
     * @return a Pair<DoubleMatrix> object <code>data</code>, where <code>data.first</code> is the feature values (X) of all instances, and the  <code>data.second</code> is the labels (Y).
     */
    public Pair<DoubleMatrix> getAllData(){
        return new Pair<DoubleMatrix>(new DoubleMatrix(this.X), new DoubleMatrix(this.Y));
    }

    /**
     * Get all feature values (X)
     * @return a matrix
     */
    public double[][] getX(){
        return this.X;
    }

    /**
     * Get all labels (Y)
     * @return a matrix
     */
    public double[][] getY(){
        return this.Y;
    }
    
    /**
     * Shuffle the ordering of instances in the dataset. Should be called before each epoch during training.
     */
    public void shuffle() {
    	this.shuffle(new Random());
    }

    /**
     * Get a minibatch of size batchsize
     * @param batchsize 
     * @return a pair of X (feature values) and Y (labels)
     */
    public Pair<DoubleMatrix> getNextMiniBatch(int batchsize) {
    	if (this.currIndex >= this.getSize()) {
            this.currIndex = 0;
            return null;
        }
    	
        int start = this.currIndex;
        int end = Math.min(start + batchsize, this.getSize());
        this.currIndex = end;
      
        double[][] bX = new double[end - start][];
        double[][] bY = new double[end - start][];
        for (int i = start; i < end; i++) {
            bX[i - start] = X[i];
            bY[i - start] = Y[i];
        }        

        return new Pair<DoubleMatrix>(
                new DoubleMatrix(bX),
                new DoubleMatrix(bY)
        );
    }

}
