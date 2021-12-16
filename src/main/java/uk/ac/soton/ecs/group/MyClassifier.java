package uk.ac.soton.ecs.group;

import java.util.ArrayList;

import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.image.FImage;
import org.openimaj.ml.annotation.Annotator;
import org.openimaj.ml.annotation.ScoredAnnotation;


/**
 * Parent class to define the structure of the individual classifiers.
 * 
 * @author Charles Powell (cp6g18)
 * @author Dzhani Daud (dsd1u19)
 */
public abstract class MyClassifier {
    
    //////////////
    // TRAINING //
    //////////////

    /**
     * Fits the classifier to the given training data.
     * 
     * @param trainingData The training data the classifier will
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
    public ArrayList<Tuple<String, String>> makeGuesses(VFSListDataset<FImage> dataset) {
        // object to store guesses
        ArrayList<Tuple<String,String>> guesses = new ArrayList<>();

        // count to keep track of image index
        int i = 0;

        // iterating through images in dataset
        for(FImage image : dataset){
            String bestAnnotation = "";
            float bestPrediction = 0.0f;

            // finding best annotation for this image
            for(Object score : this.getClassifier().annotate(image)){
                ScoredAnnotation s = (ScoredAnnotation) score;
                if(((ScoredAnnotation<?>) score).confidence > bestPrediction){
                    bestPrediction = ((ScoredAnnotation<?>) score).confidence;
                    bestAnnotation = ((ScoredAnnotation<?>) score).annotation.toString();
                }
            }
            // getting image name (filename) - ID of the form "testing/<filename>" and we need just <filename>
            String imageName = dataset.getID(i).split("/")[1];

            // adding guess to guesses list
            guesses.add(new Tuple<>(imageName, bestAnnotation));

            // incrementing count
            i++;
        }

        // returning the guesses
        return guesses;
    }

    /////////////////////////
    // GETTERS AND SETTERS //
    /////////////////////////

    public abstract Annotator getClassifier();
}