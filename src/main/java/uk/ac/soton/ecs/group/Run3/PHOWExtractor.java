package uk.ac.soton.ecs.group.Run3;

import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.dense.gradient.dsift.PyramidDenseSIFT;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.PyramidSpatialAggregator;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.util.pair.IntFloatPair;

/**
 * COMP3204: Computer Vision
 * Coursework 3
 * A Pyramid Histogram of Words Extractor
 *
 * @author Dzhem Kavak (dtk1u19)
 * @author Velimir Anastasov (vna1u19)
 * @author Konrad Sobczak (kks1g19)
 */
public class PHOWExtractor implements FeatureExtractor<DoubleFV, FImage> {

    // member variables
    PyramidDenseSIFT<FImage> pdsift;
    HardAssigner<byte[], float[], IntFloatPair> assigner;

    //////////////////
    // INITIALIZING //
    //////////////////

    /**
     * Class constructor.
     * 
     * @param pdsift pyramid dense shift
     * @param assigner hard assigner
     */
    public PHOWExtractor(
            PyramidDenseSIFT<FImage> pdsift, HardAssigner<byte[], float[], IntFloatPair> assigner) {
        this.pdsift = pdsift;
        this.assigner = assigner;
    }

    ////////////////////////
    // FEATURE EXTRACTION //
    ////////////////////////

    /**
     * Extract the features using BOVW and Pyramid Spatial Aggregator
     *
     * @param image the image of which the features have to be extracted.
     * @return The extracted features
     */
    public DoubleFV extractFeature(FImage image) {
        pdsift.analyseImage(image);

        BagOfVisualWords<byte[]> bovw = new BagOfVisualWords<>(assigner);
        PyramidSpatialAggregator<byte[], SparseIntFV> spatial = new PyramidSpatialAggregator<>(bovw, 2, 4);

        return spatial.aggregate(pdsift.getByteKeypoints(0.015f), image.getBounds()).normaliseFV();
    }
}