package uk.ac.soton.ecs.group.Run1;

import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.FloatFV;
import org.openimaj.image.FImage;
import org.openimaj.image.processing.resize.ResizeProcessor;
import org.openimaj.math.geometry.point.Point2d;
import org.openimaj.math.geometry.shape.Rectangle;

/**
 * Tiny Image Feature Extractor.
 * 
 * Extracts an feature vector equal to the tiny image representation
 * of the image.
 * 
 * @author Charles Powell (cp6g18)
 */
public class TinyImageFeatureExtractor implements FeatureExtractor<FloatFV, FImage> {

    // member variables
    int res;

    /**
     * Class constructor.
     * 
     * @param res The resolution of the tiny image (res x res pixels).
     */
    public TinyImageFeatureExtractor(int res){
        // initializing
        this.res = res;
    }
    
    /**
     * Extracts the "tiny image" feature from the image.
     * 
     * Crops the image to a fixed size (width x width), resizes the image
     * to the given resolution and concatenates the rows of the image into 
     * a single vector.
     * 
     * @param FImage The image for which the "tiny image" feature is being
     * extracted.
     */
    @Override
    public FloatFV extractFeature(FImage image) {
        // getting center of image
        Rectangle bounds = image.getBounds();
        Point2d center = bounds.calculateCentroid();

        // cropping image to square (new image is width x width)
        FImage croppedImage = image.extractCenter((int) center.getX(),
                                                    (int) center.getY(),
                                                    (int) bounds.width,
                                                    (int) bounds.width) ;

        // resizing the image to the given resolution
        ResizeProcessor resize = new ResizeProcessor(res,res);
        FImage tinyImage = croppedImage.processInplace(resize);

        // mean centering tiny image
        //tinyImage.pixels = TinyImageExtractor.meanCenterMatrix(tinyImage.pixels); /** DOES NOT IMPROVE CLASSIFIER PERFORMANCE ? */

        // normalising
        //tinyImage.pixels = TinyImageExtractor.normaliseMatrix(tinyImage.pixels); /** CAUSES INDEX ERROR IN THE EVALUATE METHOD */

        // forming feature vector from tiny image
        float [] vector = new float[this.res*this.res];
        int vectorPlace = 0;
        for(int i = 0; i < tinyImage.getHeight(); i++){
            for(int j = 0; j < tinyImage.getWidth(); j++){
                vector[vectorPlace] = tinyImage.pixels[i][j];
                vectorPlace++;
            }
        }

        // returning feature vector
        return new FloatFV(vector);
    }

    ////////////////////
    // HELPER METHODS //
    ////////////////////

    /**
     * Mean centers the row vectors in a matrix.
     * 
     * @param matrix The matrix who's row vectors are being normalised. 
     * @return The matrix with mean centered row vectors.
     */
    public static float[][] meanCenterMatrix(float[][] matrix){
        // new matrix
        float[][] meanCenteredMatrix = new float[matrix.length][matrix[0].length];

        // calculating mean values in each dimension
        float[] meanValues = new float[matrix[0].length];
        for(int col = 0; col < matrix[0].length; col++){
            float sum = 0.0f;

            for(int row = 0; row < matrix.length; row ++){
                sum += matrix[row][col];
            }
            meanValues[col] = sum / meanValues.length;
        }

        // subtracting means
        for(int col = 0; col < matrix[0].length; col++){
            for(int row = 0; row < matrix.length; row ++){
                meanCenteredMatrix[row][col] = matrix[row][col] - meanValues[col];
            }
        }

        // returning the mean centered matrix
        return meanCenteredMatrix;
    }

    /**
     * Normalises the row vectors of a matrix by making them have unit length.
     * 
     * @param matrix The matrix who's row vectors are being normalised.
     * @return The matrix with normalised row vectors.
     */
    public static float[][] normaliseMatrix(float[][] matrix){
        // new matrix
        float[][] normalisedMatrix = new float[matrix.length][matrix[0].length];

        // iterating over matrix rows
        for(int row =0; row < matrix.length; row++){
            // calculating vector length (magnitude of row vector)
            float squaredSum = 0.0f;
            for(int col = 0; col < matrix[0].length; col++){
                squaredSum += (matrix[row][col]) * (matrix[row][col]);
            }
            float length = (float) Math.sqrt(squaredSum);

            // dividing each row vector dimension by row vector magnitude
            for(int col = 0; col < matrix[0].length; col++){
                normalisedMatrix[row][col] = matrix[row][col] / length;
            }
        }

        // returning normalised matrix
        return normalisedMatrix;
    }
}