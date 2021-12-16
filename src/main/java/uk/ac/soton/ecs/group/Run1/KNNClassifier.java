package uk.ac.soton.ecs.group.Run1;


import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.FloatFV;
import org.openimaj.feature.FloatFVComparison;
import org.openimaj.image.FImage;
import org.openimaj.image.processing.resize.ResizeProcessor;
import org.openimaj.math.geometry.point.Point2d;
import org.openimaj.math.geometry.shape.Rectangle;
import org.openimaj.ml.annotation.Annotator;
import org.openimaj.ml.annotation.basic.KNNAnnotator;

import uk.ac.soton.ecs.group.MyClassifier;

/**
 * COMP3204: Computer Vision
 * 
 * Coursework 3
 * 
 * A K-Nearest Neighbour Classifier using the "tiny image" feature.
 * 
 * @author Charles Powell (cp6g18)
 */
public class KNNClassifier extends MyClassifier{

    // member variables
    private int k;
    private KNNAnnotator knnAnnotator;

    //////////////////
    // INITIALIZING //
    //////////////////

    /**
     * Class constructor.
     * 
     * @param k The value of K for the K-Nearest Neighbour Classification
     */
    public KNNClassifier(int k, int tinyImageRes, GroupedDataset trainingDataset){
        // initializing
        this.k = k;
        FeatureExtractor<FloatFV, FImage> tinyImageExtractor = new TinyImageFeatureExtractor(tinyImageRes);
        this.knnAnnotator = KNNAnnotator.create(tinyImageExtractor, FloatFVComparison.EUCLIDEAN, this.k);
        
        // fitting the classifier to the training data
        this.fit(trainingDataset);
    }

    //////////////
    // TRAINING //
    //////////////

    /**
     * Fits the KNN Annotator to the given training data.
     * 
     * @param trainingData The training data the KNN Annotator will
     * be fit to.
     */
    public void fit(GroupedDataset trainingData){
        this.knnAnnotator.train(trainingData);
    }

    /////////////////////////
    // GETTERS AND SETTERS //
    /////////////////////////

    public Annotator getClassifier(){
        return this.knnAnnotator;
    }
}