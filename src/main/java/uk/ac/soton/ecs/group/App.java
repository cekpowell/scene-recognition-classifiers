package uk.ac.soton.ecs.group;

import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Map;

import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;

import uk.ac.soton.ecs.group.Run1.*;
import uk.ac.soton.ecs.group.Run2.*;
import uk.ac.soton.ecs.group.Run3.*;

/**
 * COMP3204: Computer Vision
 * Coursework 3
 * 
 * Application to test and run the classifiers.
 * 
 * @author Charles Powell (cp6g18)
 * @author Dzhani Daud (dsd1u19)
 */
public class App {
    
    /**
     * Main method - tests the classifiers.
     * 
     * @param args The system arguments.
     */
    public static void main(String[] args) throws Exception{

        /////////////////
        // CONFIGURING //
        /////////////////

        // LOADING DATASETS //

        // training
        VFSGroupDataset<FImage> trainingData = new VFSGroupDataset<FImage>("zip:http://comp3204.ecs.soton.ac.uk/cw/training.zip", 
                                                                           ImageUtilities.FIMAGE_READER);
        trainingData.remove("training");

        // testing
        VFSListDataset<FImage> testingData = new VFSListDataset<FImage>("zip:http://comp3204.ecs.soton.ac.uk/cw/testing.zip", 
                                                                        ImageUtilities.FIMAGE_READER);  

        // OUTPUT FILENAMES //

        String run1Filename = "run1.txt";
        String run2Filename = "run2.txt";
        String run3Filename = "run3.txt";

        /////////////////////////
        // RUNNING CLASSIFIERS //
        /////////////////////////

        /**
         * Comment out 'App.runX() calls to run only certain classifiers.'
         */

        // RUN 1 //

        System.out.println();
        System.out.println("## RUN 1 ##");
        System.out.println();

        App.run1(trainingData, testingData, run1Filename, true);

        // RUN 2 //

        System.out.println();
        System.out.println("## RUN 2 ##");
        System.out.println();

        App.run2(trainingData, testingData, run2Filename, true);

        // RUN 3 //

        System.out.println();
        System.out.println("## RUN 3 ##");
        System.out.println();

        App.run3(trainingData, testingData, run3Filename, true);
    }

    ///////////////////////////
    // RUN 1: KNN CLASSIFIER //
    ///////////////////////////

    /**
     * Method to handle the testing and running of the 'Run 1' Classifier.
     * 
     * @param trainingData The training data for the classifier.
     * @param testingData The testing data for the classifier.
     * @param outputFilename The file the classifications will be written to.
     * @param evaluate If the classifier should be evaluated or not.
     */
    public static void run1(VFSGroupDataset<FImage> trainingData, VFSListDataset<FImage> testingData, String outputFilename, boolean evaluate) throws Exception{
        
        ////////////////
        // PARAMETERS //
        ////////////////

        int k = 16; /** Value of K in the KNN algorithm */
        int tinyImageRes = 16; /** Resolution of tiny image features */
        int trainingNum = 70; /** out of 100 per class */
        int testingNum = 30;  /** out of 100 per class */

        System.out.println("Running KNN Classifier with parameters : " 
                            + "\n\tk : " + k 
                            +  "\n\tTiny Image Resolution : " + tinyImageRes);
        System.out.println();

        ///////////////////////////////////////
        // EVALUATING CLASSIFIER PERFORMANCE //
        ///////////////////////////////////////

        if(evaluate){
            System.out.println("Evaluating classifier performance with parameters : "
                                                                 + "\n\tTraining Number : " + trainingNum
                                                                 + "\n\tTesting Number : " + testingNum);

            GroupedRandomSplitter<String, FImage> splits = new GroupedRandomSplitter<>(trainingData, trainingNum, 0, testingNum);

            System.out.println("Results : ");
            System.out.println(App.evaluateClassifier(new KNNClassifier(k, tinyImageRes, splits.getTrainingDataset()), splits));
        }

        ////////////////////////////////////////
        // RUNNING CLASSIFIER ON TESTING DATA //
        ////////////////////////////////////////

        App.runClassifier(new KNNClassifier(k, tinyImageRes, trainingData), testingData, outputFilename);
    }

    ///////////////////////////////
    // RUN 2: LINEAR CLASSIFIERS //
    ///////////////////////////////

    /**
     * Method to handle the testing and running of the 'Run 2' Classifier.
     * 
     * @param trainingData The training data for the classifier.
     * @param testingData The testing data for the classifier.
     * @param outputFilename The file the classifications will be written to.
     * @param evaluate If the classifier should be evaluated or not.
     */
    public static void run2(VFSGroupDataset<FImage> trainingData, VFSListDataset<FImage> testingData, String outputFilename, boolean evaluate) throws Exception{
        
        ////////////////
        // PARAMETERS //
        ////////////////

        int k = 1200; /** Value of K in the K-Means Clustering algorithm */
        int patchSize = 8; /** Size of each DSPP patch */
        int patchEvery = 4; /** We extract a patch every 4 pixels */
        int sampleSize = 120;
        int trainingNum = 80; /** out of 100 per class */
        int testingNum = 20;  /** out of 100 per class */

        System.out.println("Running LibLinear Classifier with parameters : "
                            + "\n\tk : " + k
                            +  "\n\tpatchSize : " + patchSize
                            +  "\n\tpatchEvery : " + patchEvery
                            +  "\n\tsampleSize : " + sampleSize);
        System.out.println();

        ///////////////////////////////////////
        // EVALUATING CLASSIFIER PERFORMANCE //
        ///////////////////////////////////////

        if(evaluate){
            System.out.println("Evaluating classifier performance with parameters : "
                    + "\n\tTraining Number : " + trainingNum
                    + "\n\tTesting Number : " + testingNum);

            GroupedRandomSplitter<String, FImage> splits = new GroupedRandomSplitter<>(trainingData, trainingNum, 0, testingNum);

            System.out.println("Results : ");
            System.out.println(App.evaluateClassifier(new LibLinearClassifier(splits.getTrainingDataset(), patchSize, patchEvery, k, sampleSize), splits));
        }

        ////////////////////////////////////////
        // RUNNING CLASSIFIER ON TESTING DATA //
        ////////////////////////////////////////

        App.runClassifier(new LibLinearClassifier(trainingData, patchSize, patchEvery, k, sampleSize), testingData, outputFilename);
    }

    //////////////////////////////
    // RUN 3: CUSTOM CLASSIFIER //
    //////////////////////////////

    /**
     * Method to handle the testing and running of the 'Run 3' Classifier.
     * 
     * @param trainingData The training data for the classifier.
     * @param testingData The testing data for the classifier.
     * @param outputFilename The file the classifications will be written to.
     * @param evaluate If the classifier should be evaluated or not.
     */
    public static void run3(VFSGroupDataset<FImage> trainingData, VFSListDataset<FImage> testingData, String outputFilename, boolean evaluate) throws Exception{
        
        ////////////////
        // PARAMETERS //
        ////////////////

        int k = 150; /** Value of K in the K-Means Clustering algorithm */
        int sampleSize = 120;
        int trainingNum = 80; /** out of 100 per class */
        int testingNum = 20;  /** out of 100 per class */

        System.out.println("Running LibLinear Classifier with parameters : "
                + "\n\tk : " + k
                +  "\n\tsampleSize : " + sampleSize);
        System.out.println();

        ///////////////////////////////////////
        // EVALUATING CLASSIFIER PERFORMANCE //
        ///////////////////////////////////////

        if(evaluate) {
            System.out.println("Evaluating classifier performance with parameters : "
                    + "\n\tTraining Number : " + trainingNum
                    + "\n\tTesting Number : " + testingNum);

            GroupedRandomSplitter<String, FImage> splits = new GroupedRandomSplitter<>(trainingData, trainingNum, 0, testingNum);

            System.out.println("Results : ");
            System.out.println(App.evaluateClassifier(new NaiveBayesClassifier(splits.getTrainingDataset(), k, sampleSize), splits));
        }

        ////////////////////////////////////////
        // RUNNING CLASSIFIER ON TESTING DATA //
        ////////////////////////////////////////

        App.runClassifier(new NaiveBayesClassifier(trainingData, k, sampleSize), testingData, outputFilename);
    }

    ////////////////////
    // HELPER METHODS //
    ////////////////////

    /**
     * Evaluates the performance of the classifier on the given dataset.
     * 
     * @param splits The splits dataset the performance of the classifier is being
     * evaluated against.
     */
    public static String evaluateClassifier(MyClassifier classifier, GroupedRandomSplitter<String, FImage> splits){
        // creating an evaluator object
        ClassificationEvaluator<CMResult<String>, String, FImage> eval = new ClassificationEvaluator<>(classifier.getClassifier(),
                                                                                                       splits.getTestDataset(),
                                                                                                       new CMAnalyser<FImage, String>(CMAnalyser.Strategy.SINGLE));

        Map<FImage, ClassificationResult<String>> guesses = eval.evaluate();  // getting the guesses from the evaluator
        CMResult<String> result = eval.analyse(guesses); // analysing the accuracy of the guesses
        return result.toString(); // returning the result of the evaluation
    }

    /**
     * Runs the given classifier on the given testing data, and saves the classifications
     * to the given file.
     * 
     * @param classifier The classifier being run.
     * @param testingData The data being classified.
     * @param outputFilename The file the classifications will be written to.
     */
    public static void runClassifier(MyClassifier classifier, VFSListDataset<FImage> testingData, String outputFilename){

        // GETTING CLASSIFIER PREDICTIONS //

        System.out.println();
        System.out.println("Classsifying testing data ...");

        ArrayList<Tuple<String,String>> guesses = classifier.makeGuesses(testingData);

        System.out.println("Testing data successfully classified!");

        // WRITING CLASSIFICATIONS TO FILE //

        System.out.println();
        System.out.println("Writing classifications to file : " + outputFilename + " ...");

        try{
            App.writeGuessesToFile(guesses, outputFilename);

            System.out.println("Cassifications successfully written to file : " + outputFilename + "!");
        }
        catch(Exception e){
            System.out.println("Unable to write classifications to file!");
            e.printStackTrace();
        }

        System.out.println();
    }

    /**
     * Writes the given scene classification guesses to a a file.
     * 
     * @param guesses The classification guesses to be written to a file.
     * @param filename The name of the file the classifications are being written to.
     */
    public static void writeGuessesToFile(ArrayList<Tuple<String, String>> guesses, String filename) throws Exception{
        FileWriter fileWriter = new FileWriter(filename);
        PrintWriter printWriter = new PrintWriter(fileWriter);

        for(Tuple<String, String> guess : guesses){
            printWriter.println(guess);
            printWriter.flush();
        }
    }
}
