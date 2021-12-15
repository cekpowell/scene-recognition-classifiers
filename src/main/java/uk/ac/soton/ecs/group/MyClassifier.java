package uk.ac.soton.ecs.group;

import java.util.ArrayList;

import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.evaluation.classification.Classifier;
import org.openimaj.image.FImage;


/**
 * Parent class to define the structure of the project's classifiers.
 * 
 * @author Charles Powell (cp6g18)
 */
public abstract class MyClassifier {
    
    //////////////
    // TRAINING //
    //////////////

    /**
     * Fits the KNN Annotator to the given training data.
     * 
     * @param trainingData The training data the KNN Annotator will
     * be fit to.
     */
    public abstract void fit(GroupedDataset trainingData);

    //////////////
    // GUESSING //
    //////////////

    /**
     * Assigns classifications to the given dataset of images.
     * 
     * @param dataset The dataset of images to be classified.
     * @return A mapping of images to their estimated classification.
     */
    public abstract ArrayList<Tuple<String, String>> makeGuesses(VFSListDataset<FImage> dataset);

    /////////////////////////
    // GETTERS AND SETTERS //
    /////////////////////////

    public abstract Classifier getClassifier();
}