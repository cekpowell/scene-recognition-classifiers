package uk.ac.soton.ecs.group.Run2;

import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.util.pair.IntDoublePair;

import java.awt.*;
import java.util.ArrayList;
import java.util.List;

/**
 * COMP3204: Computer Vision
 * Coursework 3
 * 
 * A Densly Sampled Pixel Patches Feature Extractor.
 *
 * @author Dzhani Daud (dsd1u19)
 * @author Konrad Sobczak (kks1g19)
 */
class DSPPExtractor implements FeatureExtractor<DoubleFV, FImage> {

    // member variables
    private final HardAssigner<double[], double[], IntDoublePair> assigner;
    private final int patchSize;
    private final int patchEvery;

    //////////////////
    // INITIALIZING //
    //////////////////

    /**
     * Class constructor.
     * 
     * @param assigner the assigner used for BOVW
     * @param patchSize the patch size
     * @param patchEvery distance between patches
     */
    public DSPPExtractor(HardAssigner<double[], double[], IntDoublePair> assigner, int patchSize, int patchEvery) {
        this.assigner = assigner;
        this.patchSize = patchSize;
        this.patchEvery = patchEvery;
    }

    ////////////////////////
    // FEATURE EXTRACTION //
    ////////////////////////

    /**
     * Extract the features using Bag Of Visual Words
     * 
     * @param image the image of which the features have to be extracted.
     *
     * @return The extracted features
     */
    public DoubleFV extractFeature(FImage image) {
        BagOfVisualWords<double[]> bovw = new BagOfVisualWords<>(assigner);
        return bovw.aggregateVectorsRaw(getPatches(image, patchSize, patchEvery)).asDoubleFV();
    }

    /**
     * Get the patches out of an image
     * 
     * @param image image from which to get the patches.
     * @param patchSize the patch size.
     * @param sampleEvery distance between patches.
     *
     * @return list of patches
     */
    public static List<double[]> getPatches(FImage image, int patchSize, int sampleEvery) {
        List<double[]> features = new ArrayList<>();

        final int patchesWidth = (int) Math.floor((image.getWidth() - patchSize) / sampleEvery) + 1;
        final int patchesHeight = (int) Math.floor((image.getHeight() - patchSize) / sampleEvery) + 1;

        FImage[][] patches = new FImage[patchesWidth][patchesHeight];

        //loop over the patches
        for (int row = 0; row < patches.length; row++) {
            for (int col = 0; col < patches[row].length; col++) {
                Point bottomRight = new Point(row * sampleEvery + patchSize, col * sampleEvery + patchSize);

                float[][] patch = new float[patchSize][patchSize];

                for (int y = bottomRight.y - patchSize; y < bottomRight.y; y++) {
                    for (int x = bottomRight.x - patchSize; x < bottomRight.x; x++) {
                        int cropX = x - (bottomRight.x - patchSize);
                        int cropY = y - (bottomRight.y - patchSize);

                        patch[cropY][cropX] = image.pixels[y][x];
                    }
                }
                features.add(normaliseAndMeanCenter(new FImage(patch).getDoublePixelVector()));
            }
        }
        return features;
    }

    ////////////////////
    // HELPER METHODS //
    ////////////////////

    /**
     * Normalise and mean center a vector.
     * 
     * @param vector vector to be mean centered and normalised.
     *
     * @return normalised vector.
     */
    public static double[] normaliseAndMeanCenter(double[] vector) {
        double length = 0;
        double mean = 0;
        for(double val : vector) mean += val;
        mean /= vector.length;

        for (int i=0; i< vector.length; i++) {
            vector[i] -= mean;
            length += Math.pow(vector[i], 2);
        }
        length = Math.sqrt(length);
        for (int i=0; i< vector.length; i++) vector[i] /= length;
        return vector;
    }
}
